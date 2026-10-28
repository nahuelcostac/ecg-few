"""Direct clinical Brugada CNN LOOCV experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from ecg_few.cnn.data import PreprocessConfig, compute_preprocess_config, load_image_tensor
from ecg_few.cnn.pipeline import (
    existing_keys,
    plot_confusion_matrix,
    resolve_device,
    set_seed,
)
from ecg_few.evaluation.metrics import brugada_metrics
from ecg_few.loocv import (
    DEFAULT_K_VALUES,
    DEFAULT_SEEDS,
    BrugadaImageRow,
    PatientRecord,
    parse_int_list,
    patients_from_rows,
    read_jsonl,
    read_manifest,
    resolve_folds_path,
    rows_for_patient_ids,
    selection_for,
    validate_fold_plan,
)

DEFAULT_OUTPUT_ROOT = Path("outputs/work/cnn_clinical_brugada_loocv")
DEFAULT_K_VALUES_TEXT = ",".join(str(value) for value in DEFAULT_K_VALUES)
DEFAULT_SEEDS_TEXT = ",".join(str(value) for value in DEFAULT_SEEDS)
CONDITIONS = ("estandar", "balanced")
BASE_SEED_OFFSET = 100_003


class ClinicalImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        rows: Sequence[BrugadaImageRow],
        dataset_root: Path,
        preprocess: PreprocessConfig,
    ) -> None:
        self.rows = list(rows)
        self.dataset_root = dataset_root
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        if row.clinical_brugada is None:
            raise ValueError(f"Clinical label is missing for {row.patient_id}.")
        image = load_image_tensor(
            self.dataset_root / row.image_path,
            image_size=self.preprocess.image_size,
            mean=self.preprocess.mean,
            std=self.preprocess.std,
        )
        label = torch.tensor(float(row.clinical_brugada), dtype=torch.float32)
        return image, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct clinical Brugada CNN LOOCV on HUCA V1 images."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/brugada_huca"))
    parser.add_argument("--folds", type=Path, default=Path(""))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--k-values", default=DEFAULT_K_VALUES_TEXT)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS_TEXT)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold-strategy",
        choices=("fixed", "val_derived_balanced_accuracy", "val_derived_f1"),
        default="val_derived_balanced_accuracy",
    )
    parser.add_argument("--loss-pos-weight", choices=("none", "auto"), default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resnet-weights", choices=("default", "none"), default="default")
    parser.add_argument("--limit-folds", type=int, default=0)
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument("--gradcam-count", type=int, default=6)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def main() -> None:
    run_with_args(parse_args())


def run_with_args(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    rows = [
        row
        for row in read_manifest(dataset_root / "labels" / "all_labels.csv")
        if row.lead.upper() == "V1"
    ]
    validate_clinical_rows(rows)
    folds = read_jsonl(resolve_folds_path(dataset_root, args.folds).resolve())
    k_values = parse_int_list(str(args.k_values))
    seeds = parse_int_list(str(args.seeds))
    conditions = parse_conditions(str(args.conditions))
    validate_fold_plan(folds, rows, k_values=k_values, seeds=seeds)
    if int(args.limit_folds) > 0:
        folds = folds[: int(args.limit_folds)]
    device = resolve_device(str(args.device))
    print(f"[INFO] Clinical CNN device: {device}")
    write_campaign_metadata(
        output_root=output_root,
        dataset_root=dataset_root,
        rows=rows,
        folds=folds,
        k_values=k_values,
        seeds=seeds,
        conditions=conditions,
        args=args,
        device=device,
    )
    for condition in conditions:
        report_dir = output_root / "reports" / condition
        summary_rows: list[dict[str, object]] = []
        for k in k_values:
            for seed in seeds:
                run_dir = output_root / "runs" / condition / "resnet18" / f"k{k}_seed{seed}"
                predictions = run_k_seed(
                    rows=rows,
                    folds=folds,
                    dataset_root=dataset_root,
                    run_dir=run_dir,
                    k=k,
                    seed=seed,
                    condition=condition,
                    args=args,
                    device=device,
                )
                summary_rows.append(
                    summarize_predictions(
                        predictions,
                        k=k,
                        seed=seed,
                        condition=condition,
                        run_dir=run_dir,
                        report_dir=report_dir,
                    )
                )
        write_summary_reports(summary_rows, report_dir)
    render_campaign_figures(output_root)


def parse_conditions(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    unknown = sorted(set(values) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unsupported clinical CNN conditions: {unknown}")
    if not values:
        raise ValueError("At least one clinical CNN condition is required.")
    return list(dict.fromkeys(values))


def validate_clinical_rows(rows: Sequence[BrugadaImageRow]) -> None:
    if not rows:
        raise ValueError("The clinical CNN dataset contains no V1 rows.")
    if any(row.clinical_brugada is None for row in rows):
        raise ValueError("Every clinical CNN row must have clinical_brugada.")
    patient_ids = {row.patient_id for row in rows}
    if len(patient_ids) != len(rows):
        raise ValueError("The clinical CNN requires exactly one V1 image per patient.")
    counts = {label: sum(int(row.clinical_brugada == label) for row in rows) for label in (0, 1)}
    if counts != {0: 201, 1: 116}:
        raise ValueError(f"Unexpected HUCA V1 clinical counts: {counts}")


def run_k_seed(
    *,
    rows: Sequence[BrugadaImageRow],
    folds: Sequence[dict[str, Any]],
    dataset_root: Path,
    run_dir: Path,
    k: int,
    seed: int,
    condition: str,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "fold_predictions.jsonl"
    resume = bool(getattr(args, "resume", True))
    existing = existing_keys(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()
    predictions = load_jsonl(output_path)
    save_json(
        run_dir / "run_config.json",
        {
            "architecture": "resnet18_binary_clinical_brugada",
            "task": "clinical_brugada",
            "condition": condition,
            "k": k,
            "seed": seed,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "threshold": float(args.threshold),
            "threshold_strategy": str(args.threshold_strategy),
            "loss_pos_weight": str(args.loss_pos_weight),
            "resnet_weights": str(args.resnet_weights),
            "image_size": int(args.image_size),
            "dataset_policy": "HUCA V1 only, clinical_brugada labels, one image per patient",
            "train_policy": "selected clinical context patients only",
            "validation_policy": "fold validation patients only",
            "test_policy": "one held-out patient per LOOCV fold",
        },
    )
    for fold in folds:
        fold_id = int(fold["fold_id"])
        test_patient_id = str(fold["test_patient_id"])
        key = f"{condition}:{fold_id}:{test_patient_id}:{k}:{seed}"
        if key in existing:
            continue
        selection = selection_for_condition(
            fold=fold,
            rows=rows,
            test_patient_id=test_patient_id,
            k=k,
            seed=seed,
            condition=condition,
            fold_id=fold_id,
        )
        context_ids = selection["context_patient_ids"]
        validation_ids = selection["validation_patient_ids"]
        assert_no_leakage(test_patient_id, context_ids, validation_ids)
        train_rows = rows_for_patient_ids(rows, context_ids)
        val_rows = rows_for_patient_ids(rows, validation_ids)
        test_rows = rows_for_patient_ids(rows, [test_patient_id])
        if len(train_rows) != k or len(test_rows) != 1:
            raise ValueError(
                f"Fold {fold_id} expected {k} train and one test row, "
                f"got {len(train_rows)} and {len(test_rows)}."
            )
        print(
            f"[INFO] Clinical CNN condition={condition} k={k} seed={seed} "
            f"fold={fold_id} test_patient={test_patient_id}"
        )
        result = train_and_predict_fold(
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            dataset_root=dataset_root,
            fold_dir=run_dir / f"fold_{fold_id:04d}",
            fold_id=fold_id,
            seed=seed,
            args=args,
            device=device,
            write_gradcam=fold_id < int(args.gradcam_count),
        )
        result.update(
            {
                "key": key,
                "fold_id": fold_id,
                "k": k,
                "seed": seed,
                "condition": condition,
                "context_patient_ids": "|".join(context_ids),
                "validation_patient_ids": "|".join(validation_ids),
            }
        )
        append_jsonl(output_path, result)
        predictions.append(result)
    write_csv(run_dir / "fold_predictions.csv", predictions)
    return predictions


def selection_for_condition(
    *,
    fold: dict[str, Any],
    rows: Sequence[BrugadaImageRow],
    test_patient_id: str,
    k: int,
    seed: int,
    condition: str,
    fold_id: int,
) -> dict[str, list[str]]:
    base = selection_for(fold, k=k, seed=seed)
    if condition == "estandar":
        return base
    patients = patients_from_rows(rows)
    validation_ids = base["validation_patient_ids"]
    context_ids = select_balanced_clinical_context_patient_ids(
        patients,
        test_patient_id=test_patient_id,
        excluded_patient_ids=validation_ids,
        k=k,
        seed=seed + fold_id * BASE_SEED_OFFSET,
    )
    return {
        "context_patient_ids": context_ids,
        "validation_patient_ids": validation_ids,
    }


def select_balanced_clinical_context_patient_ids(
    patients: Sequence[PatientRecord],
    *,
    test_patient_id: str,
    excluded_patient_ids: Sequence[str] = (),
    k: int,
    seed: int,
) -> list[str]:
    excluded = {str(test_patient_id), *(str(item) for item in excluded_patient_ids)}
    available = [patient for patient in patients if patient.patient_id not in excluded]
    rng = random.Random(seed)
    by_label = {
        label: [patient for patient in available if patient.reference_brugada == label]
        for label in (0, 1)
    }
    for candidates in by_label.values():
        candidates.sort(key=lambda patient: patient.patient_id)
        rng.shuffle(candidates)
    n_positive = k // 2
    n_negative = k - n_positive
    selected = by_label[1][:n_positive] + by_label[0][:n_negative]
    selected_ids = {patient.patient_id for patient in selected}
    if len(selected) < k:
        remaining = [patient for patient in available if patient.patient_id not in selected_ids]
        remaining.sort(key=lambda patient: patient.patient_id)
        rng.shuffle(remaining)
        selected.extend(remaining[: k - len(selected)])
    if k >= 2 and {patient.reference_brugada for patient in selected} != {0, 1}:
        raise ValueError(f"Balanced selection failed for k={k} and fold test={test_patient_id}.")
    return [patient.patient_id for patient in selected[:k]]


def assert_no_leakage(
    test_patient_id: str, context_ids: Sequence[str], validation_ids: Sequence[str]
) -> None:
    context_set = set(context_ids)
    validation_set = set(validation_ids)
    if test_patient_id in context_set or test_patient_id in validation_set:
        raise ValueError(f"Test patient leaked into train/validation: {test_patient_id}")
    if context_set.intersection(validation_set):
        raise ValueError("Training and validation patients overlap.")
    if len(context_ids) != len(context_set):
        raise ValueError("Training context contains duplicate patients.")


def train_and_predict_fold(
    *,
    train_rows: Sequence[BrugadaImageRow],
    val_rows: Sequence[BrugadaImageRow],
    test_rows: Sequence[BrugadaImageRow],
    dataset_root: Path,
    fold_dir: Path,
    fold_id: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    write_gradcam: bool,
) -> dict[str, object]:
    set_seed(seed + fold_id * BASE_SEED_OFFSET)
    fold_dir.mkdir(parents=True, exist_ok=True)
    preprocess = compute_preprocess_config(
        train_rows,
        dataset_root=dataset_root,
        image_size=int(args.image_size),
    )
    train_loader = make_clinical_loader(
        train_rows,
        dataset_root=dataset_root,
        preprocess=preprocess,
        batch_size=int(args.batch_size),
        shuffle=True,
        seed=seed + fold_id * BASE_SEED_OFFSET,
    )
    model = build_binary_model(str(args.resnet_weights)).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=clinical_pos_weight(train_rows).to(device)
        if str(args.loss_pos_weight) == "auto"
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        train_loss = run_train_epoch(model, train_loader, criterion, optimizer, device)
        val_prob = predict_probabilities(model, val_rows, dataset_root, preprocess, device)
        val_threshold = select_threshold(
            val_rows,
            val_prob,
            strategy=str(args.threshold_strategy),
            default_threshold=float(args.threshold),
        )
        val_score = score_threshold(val_rows, val_prob, val_threshold, metric="balanced_accuracy")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_threshold": val_threshold,
                "val_balanced_accuracy": val_score,
            }
        )
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= int(args.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - started
    if bool(args.save_fold_models):
        torch.save(model.state_dict(), fold_dir / "model.pt")
    save_json(fold_dir / "preprocess.json", asdict(preprocess))
    write_csv(fold_dir / "history.csv", history)
    val_prob = predict_probabilities(model, val_rows, dataset_root, preprocess, device)
    threshold = select_threshold(
        val_rows,
        val_prob,
        strategy=str(args.threshold_strategy),
        default_threshold=float(args.threshold),
    )
    threshold_source = (
        "validation" if val_rows and str(args.threshold_strategy) != "fixed" else "fixed"
    )
    test_prob = predict_probabilities(model, test_rows, dataset_root, preprocess, device)
    pred_label = int(test_prob[0] >= threshold)
    true_label = int(test_rows[0].clinical_brugada)
    if write_gradcam:
        save_binary_gradcam_panel(
            model=model,
            rows=test_rows,
            dataset_root=dataset_root,
            preprocess=preprocess,
            device=device,
            output_path=fold_dir / "gradcam_panel.png",
            title=f"Clinical CNN fold {fold_id}",
        )
    return {
        "patient_id": test_rows[0].patient_id,
        "true_label": true_label,
        "pred_label": pred_label,
        "prob_clinical_brugada": float(test_prob[0]),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "num_train_patients": len(train_rows),
        "num_train_images": len(train_rows),
        "num_val_patients": len(val_rows),
        "num_val_images": len(val_rows),
        "gradcam_written": int(write_gradcam),
    }


def build_binary_model(resnet_weights: str) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if resnet_weights == "default" else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def make_clinical_loader(
    rows: Sequence[BrugadaImageRow],
    *,
    dataset_root: Path,
    preprocess: PreprocessConfig,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ClinicalImageDataset(rows, dataset_root, preprocess),
        batch_size=max(1, batch_size),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
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


def select_threshold(
    rows: Sequence[BrugadaImageRow],
    probabilities: np.ndarray,
    *,
    strategy: str,
    default_threshold: float,
) -> float:
    if (
        strategy == "fixed"
        or not rows
        or np.unique([int(row.clinical_brugada) for row in rows]).size < 2
    ):
        return default_threshold
    candidates = np.unique(
        np.concatenate(
            [
                probabilities.reshape(-1),
                np.linspace(0.05, 0.95, 19),
                np.asarray([default_threshold]),
            ]
        )
    )
    metric = "f1" if strategy == "val_derived_f1" else "balanced_accuracy"
    best_threshold = default_threshold
    best_score = -math.inf
    for candidate in candidates:
        score = score_threshold(rows, probabilities, float(candidate), metric=metric)
        if score > best_score:
            best_score = score
            best_threshold = float(candidate)
    return best_threshold


def score_threshold(
    rows: Sequence[BrugadaImageRow],
    probabilities: np.ndarray,
    threshold: float,
    *,
    metric: str,
) -> float:
    y_true = [int(row.clinical_brugada) for row in rows]
    y_pred = [int(value >= threshold) for value in probabilities]
    result = brugada_metrics(y_true, y_pred).get(metric)
    return 0.0 if result is None else float(result)


def summarize_predictions(
    predictions: Sequence[dict[str, object]],
    *,
    k: int,
    seed: int,
    condition: str,
    run_dir: Path,
    report_dir: Path,
) -> dict[str, object]:
    y_true = np.asarray([int(row["true_label"]) for row in predictions], dtype=np.int64)
    y_pred = np.asarray([int(row["pred_label"]) for row in predictions], dtype=np.int64)
    y_prob = np.asarray(
        [float(row["prob_clinical_brugada"]) for row in predictions], dtype=np.float64
    )
    metrics = brugada_metrics(y_true, y_pred)
    probability = probability_metrics(
        y_true, y_prob, report_dir, k=k, seed=seed, condition=condition
    )
    plot_confusion_matrix(run_dir / "confusion_matrix.png", metrics["counts"])
    save_json(run_dir / "metrics.json", {"metrics": metrics, **probability})
    return {
        "model_family": "cnn_clinical",
        "model": "resnet18",
        "task": "clinical_brugada",
        "condition": condition,
        "k": k,
        "seed": seed,
        "n_patients": len(predictions),
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "sensitivity": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "tp": metrics["counts"]["tp"],
        "tn": metrics["counts"]["tn"],
        "fp": metrics["counts"]["fp"],
        "fn": metrics["counts"]["fn"],
        "roc_auc": probability["roc_auc"],
        "average_precision": probability["average_precision"],
        "predictions": (run_dir / "fold_predictions.csv").as_posix(),
        "output_dir": run_dir.as_posix(),
    }


def probability_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    report_dir: Path,
    *,
    k: int,
    seed: int,
    condition: str,
) -> dict[str, float | None]:
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = None
    try:
        average_precision = float(average_precision_score(y_true, y_prob))
    except ValueError:
        average_precision = None
    curves_dir = report_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    save_curve(
        curves_dir / f"cnn_clinical_{condition}_k{k}_seed{seed}_roc.png",
        y_true,
        y_prob,
        curve="roc",
        title=f"CNN clínica, {condition}, K={k}, semilla={seed}",
    )
    save_curve(
        curves_dir / f"cnn_clinical_{condition}_k{k}_seed{seed}_pr.png",
        y_true,
        y_prob,
        curve="pr",
        title=f"CNN clínica, {condition}, K={k}, semilla={seed}",
    )
    return {"roc_auc": roc_auc, "average_precision": average_precision}


def save_curve(
    path: Path, y_true: np.ndarray, y_prob: np.ndarray, *, curve: str, title: str
) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 3.5))
    if curve == "roc":
        order = np.argsort(-y_prob)
        sorted_true = y_true[order]
        positives = max(1, int(sorted_true.sum()))
        negatives = max(1, int(len(sorted_true) - sorted_true.sum()))
        tpr = np.cumsum(sorted_true) / positives
        fpr = np.cumsum(1 - sorted_true) / negatives
        ax.plot(np.r_[0.0, fpr], np.r_[0.0, tpr], color="#005BBB")
        ax.plot([0, 1], [0, 1], linestyle="--", color="#888888", linewidth=0.8)
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Sensibilidad")
    else:
        order = np.argsort(-y_prob)
        sorted_true = y_true[order]
        cumulative = np.cumsum(sorted_true)
        recall = cumulative / max(1, int(sorted_true.sum()))
        precision = cumulative / np.arange(1, len(sorted_true) + 1)
        ax.plot(np.r_[0.0, recall], np.r_[1.0, precision], color="#D71920")
        ax.set_xlabel("Sensibilidad")
        ax.set_ylabel("Precisión")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linewidth=0.45, alpha=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_summary_reports(rows: Sequence[dict[str, object]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "cnn_clinical_summary_by_seed.csv", rows)
    save_json(report_dir / "cnn_clinical_summary_by_seed.json", list(rows))
    grouped = aggregate_by_k(rows)
    write_csv(report_dir / "cnn_clinical_summary_by_k.csv", grouped)
    save_json(report_dir / "cnn_clinical_summary_by_k.json", grouped)
    write_confusion_by_k(report_dir / "confusion_by_k", rows)


def aggregate_by_k(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["k"]), []).append(dict(row))
    metrics = (
        "accuracy",
        "balanced_accuracy",
        "f1",
        "sensitivity",
        "specificity",
        "precision",
        "roc_auc",
        "average_precision",
    )
    result: list[dict[str, object]] = []
    for k, group in sorted(grouped.items()):
        payload: dict[str, object] = {
            "model_family": "cnn_clinical",
            "model": "resnet18",
            "task": "clinical_brugada",
            "condition": group[0]["condition"],
            "k": k,
            "n_runs": len(group),
            "n_patients": group[0]["n_patients"],
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric) not in (None, "")]
            payload[f"{metric}_mean"] = float(np.mean(values)) if values else ""
            payload[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        for count in ("tp", "tn", "fp", "fn"):
            payload[count] = int(sum(int(row[count]) for row in group))
        result.append(payload)
    return result


def write_confusion_by_k(output_dir: Path, rows: Sequence[dict[str, object]]) -> None:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["k"]), []).append(dict(row))
    for k, group in sorted(grouped.items()):
        counts = {name: sum(int(row[name]) for row in group) for name in ("tp", "tn", "fp", "fn")}
        plot_confusion_matrix(output_dir / f"cnn_clinical_k{k}_confusion_matrix.png", counts)


def render_campaign_figures(output_root: Path) -> None:
    figure_dir = output_root / "tfg_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows_by_condition = {
        condition: read_csv(output_root / "reports" / condition / "cnn_clinical_summary_by_k.csv")
        for condition in CONDITIONS
        if (output_root / "reports" / condition / "cnn_clinical_summary_by_k.csv").exists()
    }
    if not rows_by_condition:
        return
    plot_metric_conditions(
        figure_dir / "cnn_clinical_ba_by_k",
        rows_by_condition,
        "balanced_accuracy",
        "Balanced Accuracy (BA)",
    )
    plot_metric_conditions(figure_dir / "cnn_clinical_f1_by_k", rows_by_condition, "f1", "F1")
    plot_sensitivity_specificity_conditions(
        figure_dir / "cnn_clinical_sens_spec_by_k",
        rows_by_condition,
    )


def plot_metric_conditions(
    base_path: Path, rows_by_condition: dict[str, list[dict[str, str]]], metric: str, ylabel: str
) -> None:
    colors = {"estandar": "#005BBB", "balanced": "#D71920"}
    labels = {"estandar": "CNN clínica estándar", "balanced": "CNN clínica balanceada"}
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    for condition, rows in rows_by_condition.items():
        rows = sorted(rows, key=lambda row: int(row["k"]))
        x = [int(row["k"]) for row in rows]
        mean = [float(row[f"{metric}_mean"]) for row in rows]
        std = [float(row[f"{metric}_std"]) for row in rows]
        ax.errorbar(
            x,
            mean,
            yerr=std,
            color=colors[condition],
            marker="o",
            linewidth=1.15,
            capsize=2.5,
            label=labels[condition],
        )
    ax.set_xlabel(r"Número de ejemplos $K$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.grid(True, linewidth=0.45, alpha=0.7)
    ax.legend(fontsize=7, frameon=True)
    fig.tight_layout()
    for suffix, dpi in ((".pdf", 300), (".png", 300)):
        fig.savefig(base_path.with_suffix(suffix), dpi=dpi)
    plt.close(fig)


def plot_sensitivity_specificity_conditions(
    base_path: Path,
    rows_by_condition: dict[str, list[dict[str, str]]],
) -> None:
    colors = {
        ("estandar", "sensitivity"): "#005BBB",
        ("estandar", "specificity"): "#6FA8E8",
        ("balanced", "sensitivity"): "#D71920",
        ("balanced", "specificity"): "#F0A53A",
    }
    labels = {
        ("estandar", "sensitivity"): "CNN estándar, sensibilidad",
        ("estandar", "specificity"): "CNN estándar, especificidad",
        ("balanced", "sensitivity"): "CNN balanceada, sensibilidad",
        ("balanced", "specificity"): "CNN balanceada, especificidad",
    }
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for condition, rows in rows_by_condition.items():
        rows = sorted(rows, key=lambda row: int(row["k"]))
        x = [int(row["k"]) for row in rows]
        for metric in ("sensitivity", "specificity"):
            mean = [float(row[f"{metric}_mean"]) for row in rows]
            std = [float(row[f"{metric}_std"]) for row in rows]
            ax.errorbar(
                x,
                mean,
                yerr=std,
                color=colors[(condition, metric)],
                marker="o" if metric == "sensitivity" else "^",
                markersize=3.0,
                linewidth=1.0,
                capsize=2.0,
                label=labels[(condition, metric)],
            )
    ax.set_xlabel(r"Número de ejemplos $K$")
    ax.set_ylabel("Proporción")
    ax.set_ylim(0, 1.02)
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.grid(True, linewidth=0.45, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=True, ncol=2)
    fig.tight_layout()
    for suffix, dpi in ((".pdf", 300), (".png", 300)):
        fig.savefig(base_path.with_suffix(suffix), dpi=dpi)
    plt.close(fig)


def write_campaign_metadata(
    *,
    output_root: Path,
    dataset_root: Path,
    rows: Sequence[BrugadaImageRow],
    folds: Sequence[dict[str, Any]],
    k_values: Sequence[int],
    seeds: Sequence[int],
    conditions: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    save_json(
        output_root / "campaign_manifest.json",
        {
            "campaign": "cnn_clinical_brugada_loocv",
            "status": "provisional",
            "dataset_root": dataset_root.as_posix(),
            "dataset_rows_v1": len(rows),
            "clinical_label_counts": {"0": 201, "1": 116},
            "folds_used": len(folds),
            "k_values": list(k_values),
            "seeds": list(seeds),
            "conditions": list(conditions),
            "device": str(device),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "command": " ".join(sys.argv),
            "config": vars(args),
        },
    )


def save_binary_gradcam_panel(
    *,
    model: nn.Module,
    rows: Sequence[BrugadaImageRow],
    dataset_root: Path,
    preprocess: PreprocessConfig,
    device: torch.device,
    output_path: Path,
    title: str,
) -> None:
    target_module = model.layer4[-1] if hasattr(model, "layer4") else None
    if target_module is None:
        return
    activations: torch.Tensor | None = None
    gradients: torch.Tensor | None = None

    def forward_hook(
        _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        nonlocal activations
        activations = output.detach()

    def backward_hook(
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        nonlocal gradients
        gradients = grad_output[0].detach()

    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    try:
        fig, axes = plt.subplots(len(rows), 3, figsize=(9.5, max(2.8, 2.8 * len(rows))))
        axes = np.atleast_2d(axes)
        for row_index, row in enumerate(rows):
            image_path = dataset_root / row.image_path
            input_tensor = (
                load_image_tensor(
                    image_path,
                    image_size=preprocess.image_size,
                    mean=preprocess.mean,
                    std=preprocess.std,
                )
                .unsqueeze(0)
                .to(device)
            )
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logit = model(input_tensor)
                logit[:, 0].sum().backward()
            if activations is None or gradients is None:
                continue
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(
                cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
            )
            heatmap = cam.squeeze().detach().cpu().numpy()
            heatmap /= max(float(heatmap.max()), 1e-8)
            display = input_tensor[0, 0].detach().cpu().numpy()
            axes[row_index, 0].imshow(display, cmap="gray")
            axes[row_index, 1].imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
            axes[row_index, 2].imshow(display, cmap="gray")
            axes[row_index, 2].imshow(heatmap, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
            for col in range(3):
                axes[row_index, col].axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
    finally:
        forward_handle.remove()
        backward_handle.remove()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def append_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
