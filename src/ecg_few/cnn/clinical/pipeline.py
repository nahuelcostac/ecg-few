"""Clinical pipeline for ECG experiments."""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ecg_few.cnn.clinical.folds import train_and_predict_fold
from ecg_few.cnn.clinical.settings import (
    CONDITIONS,
    DEFAULT_K_VALUES_TEXT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEEDS_TEXT,
)
from ecg_few.cnn.device import resolve_device
from ecg_few.configuration import parse_config_args
from ecg_few.experiment_io import (
    append_jsonl,
    existing_keys,
    prepare_campaign,
    save_json,
    validate_output_path,
    write_csv,
)
from ecg_few.experiment_io import load_existing_predictions as load_jsonl
from ecg_few.experiment_io.safety import public_config
from ecg_few.protocols.clinical import (
    assert_no_leakage,
    parse_conditions,
    selection_for_condition,
    validate_clinical_rows,
)
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
    parse_int_list,
    read_jsonl,
    read_manifest,
    resolve_folds_path,
    rows_for_patient_ids,
    validate_fold_plan,
)
from ecg_few.reporting.clinical import (
    render_campaign_figures,
    summarize_predictions,
    write_summary_reports,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    return parse_config_args(parser, argv)


def main(argv: list[str] | None = None) -> None:
    run_with_args(parse_args(argv))


def run_with_args(args: argparse.Namespace) -> None:
    validate_output_path(Path(args.output_root))
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
    prepare_campaign(
        output_root,
        vars(args),
        inputs={
            "dataset_manifest": dataset_root / "labels" / "all_labels.csv",
            "folds": resolve_folds_path(dataset_root, args.folds),
        },
        resume=bool(args.resume),
    )
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
            "clinical_label_counts": {
                str(label): sum(row.clinical_brugada == label for row in rows) for label in (0, 1)
            },
            "folds_used": len(folds),
            "k_values": list(k_values),
            "seeds": list(seeds),
            "conditions": list(conditions),
            "device": str(device),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "config": public_config(vars(args)),
        },
    )


_LEGACY_EXPORTS = {
    "ClinicalImageDataset": "ecg_few.cnn.clinical.data",
    "make_clinical_loader": "ecg_few.cnn.clinical.data",
    "build_binary_model": "ecg_few.cnn.clinical.training",
    "run_train_epoch": "ecg_few.cnn.clinical.training",
    "predict_probabilities": "ecg_few.cnn.clinical.training",
    "clinical_pos_weight": "ecg_few.cnn.clinical.training",
    "select_threshold": "ecg_few.cnn.clinical.thresholds",
    "score_threshold": "ecg_few.cnn.clinical.thresholds",
    "selection_for_condition": "ecg_few.protocols.clinical",
    "select_balanced_clinical_context_patient_ids": "ecg_few.protocols.clinical",
    "assert_no_leakage": "ecg_few.protocols.clinical",
    "validate_clinical_rows": "ecg_few.protocols.clinical",
    "parse_conditions": "ecg_few.protocols.clinical",
    "summarize_predictions": "ecg_few.reporting.clinical",
    "probability_metrics": "ecg_few.reporting.clinical",
    "save_curve": "ecg_few.reporting.clinical",
    "write_summary_reports": "ecg_few.reporting.clinical",
    "aggregate_by_k": "ecg_few.reporting.clinical",
    "write_confusion_by_k": "ecg_few.reporting.clinical",
    "render_campaign_figures": "ecg_few.reporting.clinical",
    "plot_metric_conditions": "ecg_few.reporting.clinical",
    "plot_sensitivity_specificity_conditions": "ecg_few.reporting.clinical",
    "save_binary_gradcam_panel": "ecg_few.cnn.clinical.gradcam",
    "train_and_predict_fold": "ecg_few.cnn.clinical.folds",
}


def __getattr__(name: str):
    from importlib import import_module

    if name in _LEGACY_EXPORTS:
        return getattr(import_module(_LEGACY_EXPORTS[name]), name)
    settings = import_module("ecg_few.cnn.clinical.settings")
    if hasattr(settings, name):
        return getattr(settings, name)
    raise AttributeError(name)


if __name__ == "__main__":
    main()
