"""Folds for ECG experiments."""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ecg_few.cnn.data import (
    PreprocessConfig,
    compute_preprocess_config,
)
from ecg_few.cnn.device import set_seed
from ecg_few.cnn.domain_adaptation import (
    DomainClassifier,
    copy_encoder_weights,
    pretrain_encoder_simclr,
)
from ecg_few.cnn.gradcam import save_gradcam_panel
from ecg_few.cnn.models import build_model
from ecg_few.cnn.thresholds import (
    lead_probabilities,
    patient_condition_probabilities,
    score_patient_threshold,
    select_patient_threshold,
)
from ecg_few.cnn.training import (
    evaluate_loss,
    loss_pos_weight,
    make_loader,
    make_target_loader,
    predict_rows,
    run_domain_train_epoch,
)
from ecg_few.experiment_io import save_json, write_csv
from ecg_few.findings import LABEL_NAMES, findings_to_brugada
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
)


def train_and_predict_fold(
    *,
    train_rows: Sequence[BrugadaImageRow],
    val_rows: Sequence[BrugadaImageRow],
    test_rows: Sequence[BrugadaImageRow],
    train_dataset_root: Path,
    eval_dataset_root: Path,
    target_dataset_root: Path | None,
    target_rows: Sequence[BrugadaImageRow] | None,
    fold_dir: Path,
    fold_id: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    write_gradcam: bool,
) -> dict[str, object]:
    set_seed(seed + fold_id * 100_003)
    fold_dir.mkdir(parents=True, exist_ok=True)
    preprocess = compute_preprocess_config(
        train_rows,
        dataset_root=train_dataset_root,
        image_size=int(args.image_size),
    )
    train_loader = make_loader(
        train_rows,
        dataset_root=train_dataset_root,
        preprocess=preprocess,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        seed=seed,
        device=device,
    )
    val_loader = (
        make_loader(
            val_rows,
            dataset_root=eval_dataset_root,
            preprocess=preprocess,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            seed=seed,
            device=device,
        )
        if val_rows and all(row.has_qrs_labels for row in val_rows)
        else None
    )
    model = build_model(resnet_weights=str(args.resnet_weights)).to(device)
    if target_dataset_root is not None and target_rows and int(args.ssl_pretrain_epochs) > 0:
        encoder, _ssl_payload = pretrain_encoder_simclr(
            target_rows=target_rows,
            target_dataset_root=target_dataset_root,
            preprocess=preprocess,
            run_dir=fold_dir / "ssl_pretrain",
            seed=seed + fold_id * 100_003,
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
            resnet_weights=str(args.resnet_weights),
            epochs=int(args.ssl_pretrain_epochs),
            lr=float(args.ssl_pretrain_lr),
            weight_decay=float(args.weight_decay),
            projection_dim=int(args.ssl_projection_dim),
            temperature=float(args.ssl_temperature),
        )
        if encoder is not None:
            copy_encoder_weights(model, encoder)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=loss_pos_weight(train_rows).to(device)
        if str(args.loss_pos_weight) == "auto"
        else None
    )
    domain_classifier = (
        DomainClassifier().to(device) if str(args.domain_adaptation) == "dann" else None
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters())
        + ([] if domain_classifier is None else list(domain_classifier.parameters())),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    target_loader = make_target_loader(
        target_rows=target_rows,
        target_dataset_root=target_dataset_root,
        preprocess=preprocess,
        args=args,
        seed=seed + fold_id * 100_003,
        device=device,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_score = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        epoch_payload = run_domain_train_epoch(
            model=model,
            source_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            args=args,
            epoch=epoch,
            target_loader=target_loader,
            domain_classifier=domain_classifier,
        )
        train_loss = float(epoch_payload["train_loss"])
        val_score = -train_loss
        val_loss: float | None = None
        if val_rows:
            val_payload = predict_rows(
                model,
                val_rows,
                dataset_root=eval_dataset_root,
                preprocess=preprocess,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                device=device,
            )
            if val_loader is not None:
                val_loss = evaluate_loss(model, val_loader, criterion, device)
            threshold = select_patient_threshold(
                rows=val_rows,
                y_prob=np.asarray(val_payload["y_prob"], dtype=np.float32),
                strategy=str(args.threshold_strategy),
                default_threshold=float(args.threshold),
            )
            val_score = score_patient_threshold(
                rows=val_rows,
                y_prob=np.asarray(val_payload["y_prob"], dtype=np.float32),
                threshold=threshold,
                metric="balanced_accuracy",
            )
        history.append(
            {
                "epoch": epoch,
                "val_loss": "" if val_loss is None else val_loss,
                "monitor_score": val_score,
                **epoch_payload,
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
    train_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)
    if bool(args.save_fold_models):
        torch.save(model.state_dict(), fold_dir / "model.pt")
    save_json(fold_dir / "preprocess.json", asdict(preprocess))
    write_csv(fold_dir / "history.csv", history)

    threshold = float(args.threshold)
    threshold_source = "fixed"
    if val_rows and str(args.threshold_strategy) != "fixed":
        val_payload = predict_rows(
            model,
            val_rows,
            dataset_root=eval_dataset_root,
            preprocess=preprocess,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
        )
        threshold = select_patient_threshold(
            rows=val_rows,
            y_prob=np.asarray(val_payload["y_prob"], dtype=np.float32),
            strategy=str(args.threshold_strategy),
            default_threshold=float(args.threshold),
        )
        threshold_source = "validation"

    test_payload = predict_rows(
        model,
        test_rows,
        dataset_root=eval_dataset_root,
        preprocess=preprocess,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )
    lead_probs = lead_probabilities(test_rows, np.asarray(test_payload["y_prob"], dtype=np.float32))
    patient_probs = patient_condition_probabilities(
        test_rows,
        np.asarray(test_payload["y_prob"], dtype=np.float32),
    )
    pred_findings = {label: bool(patient_probs[label] >= threshold) for label in LABEL_NAMES}
    pred_label = findings_to_brugada(pred_findings)
    true_label = int(test_rows[0].reference_brugada)
    prob_derived_brugada = float(min(patient_probs.values()))
    gradcam_written = 0
    if write_gradcam:
        save_gradcam_panel(
            model=model,
            rows=test_rows,
            dataset_root=eval_dataset_root,
            preprocess=preprocess,
            device=device,
            output_path=fold_dir / "gradcam_panel.png",
            title=f"fold {fold_id} patient {test_rows[0].patient_id}",
        )
        gradcam_written = 1
    return {
        "patient_id": test_rows[0].patient_id,
        "true_label": true_label,
        "pred_label": pred_label,
        "prob_derived_brugada": prob_derived_brugada,
        **{f"prob_{label.lower()}": patient_probs[label] for label in LABEL_NAMES},
        **{f"pred_{label.lower()}": int(pred_findings[label]) for label in LABEL_NAMES},
        "threshold": threshold,
        "threshold_source": threshold_source,
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "num_train_patients": len({row.patient_id for row in train_rows}),
        "num_train_images": len(train_rows),
        "domain_adaptation": str(args.domain_adaptation),
        "target_domain_patients": 0
        if target_rows is None
        else len({row.patient_id for row in target_rows}),
        "target_domain_images": 0 if target_rows is None else len(target_rows),
        "num_val_patients": len({row.patient_id for row in val_rows}),
        "num_val_images": len(val_rows),
        "lead_probs": json.dumps(lead_probs, sort_keys=True),
        "patient_condition_probs": json.dumps(patient_probs, sort_keys=True),
        "pred_findings": json.dumps(pred_findings, sort_keys=True),
        "gradcam_written": gradcam_written,
    }


def train_qrs_model(
    *,
    train_rows: Sequence[BrugadaImageRow],
    train_dataset_root: Path,
    target_dataset_root: Path | None,
    target_rows: Sequence[BrugadaImageRow] | None,
    run_dir: Path,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[nn.Module, PreprocessConfig, dict[str, object]]:
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    preprocess = compute_preprocess_config(
        train_rows,
        dataset_root=train_dataset_root,
        image_size=int(args.image_size),
    )
    train_loader = make_loader(
        train_rows,
        dataset_root=train_dataset_root,
        preprocess=preprocess,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        seed=seed,
        device=device,
    )
    model = build_model(resnet_weights=str(args.resnet_weights)).to(device)
    ssl_payload: dict[str, object] = {
        "ssl_pretrain_epochs": 0,
        "ssl_pretrain_seconds": 0.0,
    }
    if target_dataset_root is not None and target_rows and int(args.ssl_pretrain_epochs) > 0:
        encoder, ssl_payload = pretrain_encoder_simclr(
            target_rows=target_rows,
            target_dataset_root=target_dataset_root,
            preprocess=preprocess,
            run_dir=run_dir / "ssl_pretrain",
            seed=seed,
            image_size=int(args.image_size),
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
            resnet_weights=str(args.resnet_weights),
            epochs=int(args.ssl_pretrain_epochs),
            lr=float(args.ssl_pretrain_lr),
            weight_decay=float(args.weight_decay),
            projection_dim=int(args.ssl_projection_dim),
            temperature=float(args.ssl_temperature),
        )
        if encoder is not None:
            copy_encoder_weights(model, encoder)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=loss_pos_weight(train_rows).to(device)
        if str(args.loss_pos_weight) == "auto"
        else None
    )
    domain_classifier = (
        DomainClassifier().to(device) if str(args.domain_adaptation) == "dann" else None
    )
    optimizer = torch.optim.AdamW(
        list(model.parameters())
        + ([] if domain_classifier is None else list(domain_classifier.parameters())),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    target_loader = make_target_loader(
        target_rows=target_rows,
        target_dataset_root=target_dataset_root,
        preprocess=preprocess,
        args=args,
        seed=seed,
        device=device,
    )
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, int(args.epochs) + 1):
        epoch_payload = run_domain_train_epoch(
            model=model,
            source_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            args=args,
            epoch=epoch,
            target_loader=target_loader,
            domain_classifier=domain_classifier,
        )
        history.append({"epoch": epoch, **epoch_payload})
    train_seconds = time.perf_counter() - started
    if bool(args.save_fold_models):
        torch.save(model.state_dict(), run_dir / "model.pt")
    save_json(run_dir / "preprocess.json", asdict(preprocess))
    write_csv(run_dir / "history.csv", history)
    return (
        model,
        preprocess,
        {
            "best_epoch": int(args.epochs),
            "train_seconds": train_seconds,
            "num_train_patients": len({row.patient_id for row in train_rows}),
            "num_train_images": len(train_rows),
            "domain_adaptation": str(args.domain_adaptation),
            "target_domain_patients": 0
            if target_rows is None
            else len({row.patient_id for row in target_rows}),
            "target_domain_images": 0 if target_rows is None else len(target_rows),
            **ssl_payload,
        },
    )


def predict_fold_with_trained_model(
    *,
    model: nn.Module,
    preprocess: PreprocessConfig,
    val_rows: Sequence[BrugadaImageRow],
    test_rows: Sequence[BrugadaImageRow],
    eval_dataset_root: Path,
    fold_dir: Path,
    fold_id: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    write_gradcam: bool,
) -> dict[str, object]:
    threshold = float(args.threshold)
    threshold_source = "fixed"
    if val_rows and str(args.threshold_strategy) != "fixed":
        val_payload = predict_rows(
            model,
            val_rows,
            dataset_root=eval_dataset_root,
            preprocess=preprocess,
            batch_size=int(args.batch_size),
            num_workers=int(args.num_workers),
            device=device,
        )
        threshold = select_patient_threshold(
            rows=val_rows,
            y_prob=np.asarray(val_payload["y_prob"], dtype=np.float32),
            strategy=str(args.threshold_strategy),
            default_threshold=float(args.threshold),
        )
        threshold_source = "validation"

    test_payload = predict_rows(
        model,
        test_rows,
        dataset_root=eval_dataset_root,
        preprocess=preprocess,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        device=device,
    )
    lead_probs = lead_probabilities(test_rows, np.asarray(test_payload["y_prob"], dtype=np.float32))
    patient_probs = patient_condition_probabilities(
        test_rows,
        np.asarray(test_payload["y_prob"], dtype=np.float32),
    )
    pred_findings = {label: bool(patient_probs[label] >= threshold) for label in LABEL_NAMES}
    pred_label = findings_to_brugada(pred_findings)
    true_label = int(test_rows[0].reference_brugada)
    prob_derived_brugada = float(min(patient_probs.values()))
    gradcam_written = 0
    if write_gradcam:
        save_gradcam_panel(
            model=model,
            rows=test_rows,
            dataset_root=eval_dataset_root,
            preprocess=preprocess,
            device=device,
            output_path=fold_dir / "gradcam_panel.png",
            title=f"fold {fold_id} patient {test_rows[0].patient_id}",
        )
        gradcam_written = 1
    return {
        "patient_id": test_rows[0].patient_id,
        "true_label": true_label,
        "pred_label": pred_label,
        "prob_derived_brugada": prob_derived_brugada,
        **{f"prob_{label.lower()}": patient_probs[label] for label in LABEL_NAMES},
        **{f"pred_{label.lower()}": int(pred_findings[label]) for label in LABEL_NAMES},
        "threshold": threshold,
        "threshold_source": threshold_source,
        "num_val_patients": len({row.patient_id for row in val_rows}),
        "num_val_images": len(val_rows),
        "lead_probs": json.dumps(lead_probs, sort_keys=True),
        "patient_condition_probs": json.dumps(patient_probs, sort_keys=True),
        "pred_findings": json.dumps(pred_findings, sort_keys=True),
        "gradcam_written": gradcam_written,
    }
