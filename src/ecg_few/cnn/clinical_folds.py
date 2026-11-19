"""Clinical folds for ECG experiments."""

from __future__ import annotations

import argparse
import copy
import math
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from ecg_few.cnn.clinical_data import make_clinical_loader
from ecg_few.cnn.clinical_gradcam import save_binary_gradcam_panel
from ecg_few.cnn.clinical_settings import BASE_SEED_OFFSET
from ecg_few.cnn.clinical_thresholds import score_threshold, select_threshold
from ecg_few.cnn.clinical_training import (
    build_binary_model,
    clinical_pos_weight,
    predict_probabilities,
    run_train_epoch,
)
from ecg_few.cnn.data import compute_preprocess_config
from ecg_few.cnn.device import set_seed
from ecg_few.experiment_io import save_json, write_csv
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
)


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
