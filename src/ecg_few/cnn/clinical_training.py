"""Clinical training for ECG experiments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from ecg_few.cnn.data import PreprocessConfig, load_image_tensor
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
)


def build_binary_model(resnet_weights: str) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if resnet_weights == "default" else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


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
        labels = labels.to(device, non_blocking=True).view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(labels.shape[0])
        total += int(labels.shape[0])
    return total_loss / max(1, total)


def predict_probabilities(
    model: nn.Module,
    rows: Sequence[BrugadaImageRow],
    dataset_root: Path,
    preprocess: PreprocessConfig,
    device: torch.device,
) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=np.float64)
    model.eval()
    probabilities: list[float] = []
    with torch.no_grad():
        for start in range(0, len(rows), 32):
            batch_rows = rows[start : start + 32]
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
            ).to(device)
            batch_probabilities = torch.sigmoid(model(images)).reshape(-1).cpu().tolist()
            probabilities.extend(float(value) for value in batch_probabilities)
    return np.asarray(probabilities, dtype=np.float64)


def clinical_pos_weight(rows: Sequence[BrugadaImageRow]) -> torch.Tensor:
    positive = sum(int(row.clinical_brugada == 1) for row in rows)
    negative = len(rows) - positive
    weight = negative / positive if positive and negative else 1.0
    return torch.tensor([float(weight)], dtype=torch.float32)
