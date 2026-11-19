"""Training for ECG experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from ecg_few.cnn.data import (
    ECGImageDataset,
    PreprocessConfig,
    load_image_tensor,
)
from ecg_few.cnn.domain_adaptation import (
    DomainClassifier,
    GradientReversal,
    coral_loss,
    cycle_batches,
    domain_adaptation_weight,
    make_unlabeled_loader,
    mmd_loss,
    resnet_features,
    resnet_logits_from_features,
)
from ecg_few.findings import LABEL_NAMES
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
)


def make_loader(
    rows: Sequence[BrugadaImageRow],
    *,
    dataset_root: Path,
    preprocess: PreprocessConfig,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ECGImageDataset(rows, dataset_root, preprocess),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def make_target_loader(
    *,
    target_rows: Sequence[BrugadaImageRow] | None,
    target_dataset_root: Path | None,
    preprocess: PreprocessConfig,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
) -> DataLoader[torch.Tensor] | None:
    if str(args.domain_adaptation) == "none" or not target_rows or target_dataset_root is None:
        return None
    return make_unlabeled_loader(
        target_rows,
        dataset_root=target_dataset_root,
        preprocess=preprocess,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        seed=seed,
        device=device,
    )


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(labels.shape[0])
        total += int(labels.shape[0])
    return total_loss / max(1, total)


def run_domain_train_epoch(
    *,
    model: nn.Module,
    source_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    target_loader: DataLoader[torch.Tensor] | None,
    domain_classifier: DomainClassifier | None,
) -> dict[str, float]:
    method = str(args.domain_adaptation)
    if method == "none" or target_loader is None:
        train_loss = run_train_epoch(model, source_loader, criterion, optimizer, device)
        return {
            "train_loss": train_loss,
            "supervised_loss": train_loss,
            "adaptation_loss": 0.0,
            "domain_weight": 0.0,
        }

    model.train()
    if domain_classifier is not None:
        domain_classifier.train()
    target_iter = cycle_batches(target_loader)
    total_loss = 0.0
    total_supervised = 0.0
    total_adaptation = 0.0
    total = 0
    weight = domain_adaptation_weight(
        epoch,
        int(args.epochs),
        float(args.domain_adaptation_weight),
    )
    kernel_scales = parse_float_list(str(args.mmd_kernel_scales))
    for source_images, labels in source_loader:
        source_images = source_images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        target_images = next(target_iter).to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        source_features = resnet_features(model, source_images)
        target_features = resnet_features(model, target_images)
        logits = resnet_logits_from_features(model, source_features)
        supervised_loss = criterion(logits, labels)

        if method == "coral":
            adaptation_loss = coral_loss(source_features, target_features)
            loss = supervised_loss + weight * adaptation_loss
        elif method == "mmd":
            adaptation_loss = mmd_loss(
                source_features,
                target_features,
                kernel_scales=kernel_scales,
            )
            loss = supervised_loss + weight * adaptation_loss
        elif method == "dann":
            if domain_classifier is None:
                raise RuntimeError("DANN requires a domain classifier.")
            features = torch.cat([source_features, target_features], dim=0)
            domain_labels = torch.cat(
                [
                    torch.zeros(source_features.shape[0], device=device),
                    torch.ones(target_features.shape[0], device=device),
                ]
            )
            domain_logits = domain_classifier(GradientReversal.apply(features, weight))
            adaptation_loss = F.binary_cross_entropy_with_logits(
                domain_logits,
                domain_labels,
            )
            loss = supervised_loss + adaptation_loss
        else:
            raise ValueError(f"Unsupported domain adaptation method: {method}")

        loss.backward()
        optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_supervised += float(supervised_loss.item()) * batch_size
        total_adaptation += float(adaptation_loss.item()) * batch_size
        total += batch_size

    return {
        "train_loss": total_loss / max(1, total),
        "supervised_loss": total_supervised / max(1, total),
        "adaptation_loss": total_adaptation / max(1, total),
        "domain_weight": weight,
    }


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = criterion(model(images), labels)
            total_loss += float(loss.item()) * int(labels.shape[0])
            total += int(labels.shape[0])
    return total_loss / max(1, total)


def predict_rows(
    model: nn.Module,
    rows: Sequence[BrugadaImageRow],
    *,
    dataset_root: Path,
    preprocess: PreprocessConfig,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    _ = num_workers
    model.eval()
    y_true: list[list[int]] = []
    y_prob: list[list[float]] = []
    with torch.no_grad():
        for start in range(0, len(rows), max(1, batch_size)):
            batch_rows = list(rows[start : start + max(1, batch_size)])
            if not batch_rows:
                continue
            images = torch.stack(
                [
                    load_image_tensor(
                        dataset_root / row.image_path,
                        image_size=preprocess.image_size,
                        mean=preprocess.mean,
                        std=preprocess.std,
                    )
                    for row in batch_rows
                ]
            )
            logits = model(images.to(device, non_blocking=True))
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            y_prob.extend(probs.astype(float).tolist())
            for row in batch_rows:
                if row.has_qrs_labels:
                    y_true.append([row.findings[label_name] for label_name in LABEL_NAMES])
    return {
        "y_true": np.asarray(y_true, dtype=np.int64),
        "y_prob": np.asarray(y_prob, dtype=np.float64),
    }


def loss_pos_weight(rows: Sequence[BrugadaImageRow]) -> torch.Tensor:
    weights: list[float] = []
    for label_name in LABEL_NAMES:
        positives = sum(row.findings[label_name] for row in rows)
        negatives = len(rows) - positives
        weight = 1.0
        if positives > 0 and negatives > 0:
            weight = negatives / positives
        weights.append(float(weight))
    return torch.tensor(weights, dtype=torch.float32)


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def exclude_patient_rows(
    rows: Sequence[BrugadaImageRow] | None,
    patient_id: str,
) -> list[BrugadaImageRow] | None:
    if rows is None:
        return None
    return [row for row in rows if row.patient_id != patient_id]
