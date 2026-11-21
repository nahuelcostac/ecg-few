"""Pipeline for ECG experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from ecg_few.cnn.device import resolve_device
from ecg_few.cnn.morphology.folds import (
    predict_fold_with_trained_model,
    train_and_predict_fold,
    train_qrs_model,
)
from ecg_few.cnn.morphology.settings import DEFAULT_MODEL_NAME, ONLY_LEAD
from ecg_few.cnn.morphology.training import exclude_patient_rows
from ecg_few.configuration import parse_config_args
from ecg_few.experiment_io import (
    append_jsonl,
    existing_keys,
    load_existing_predictions,
    prepare_campaign,
    save_json,
    validate_output_path,
    write_csv,
)
from ecg_few.protocols.loocv import (
    DEFAULT_K_VALUES,
    DEFAULT_SEEDS,
    BrugadaImageRow,
    parse_int_list,
    patients_from_rows,
    read_jsonl,
    read_manifest,
    resolve_folds_path,
    rows_for_patient_ids,
    select_context_patient_ids,
    selection_for,
    validate_fold_plan,
)
from ecg_few.reporting.cnn import summarize_predictions, write_summary_reports


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CNN few-shot LOOCV for QRS findings and derived Brugada."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/brugada_huca"))
    parser.add_argument(
        "--train-dataset-root",
        type=Path,
        default=Path(""),
        help="Optional synthetic QRS dataset used for detector training instead of fold context.",
    )
    parser.add_argument(
        "--external-train-mode",
        choices=("shared", "per_fold"),
        default="shared",
        help="When using --train-dataset-root, train once per k/seed or once per real fold.",
    )
    parser.add_argument("--folds", type=Path, default=Path(""))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/work/cnn_loocv"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
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
    parser.add_argument(
        "--target-dataset-root",
        type=Path,
        default=Path(""),
        help=(
            "Optional unlabeled target-domain dataset for SSL/CORAL/MMD/DANN. "
            "Defaults to --dataset-root when --train-dataset-root is used."
        ),
    )
    parser.add_argument(
        "--domain-adaptation",
        choices=("none", "coral", "mmd", "dann"),
        default="none",
    )
    parser.add_argument("--domain-adaptation-weight", type=float, default=0.1)
    parser.add_argument(
        "--mmd-kernel-scales",
        default="0.5,1.0,2.0,4.0",
        help="Comma-separated RBF kernel scales for MMD feature alignment.",
    )
    parser.add_argument("--ssl-pretrain-epochs", type=int, default=0)
    parser.add_argument("--ssl-pretrain-lr", type=float, default=1e-4)
    parser.add_argument("--ssl-projection-dim", type=int, default=128)
    parser.add_argument("--ssl-temperature", type=float, default=0.2)
    parser.add_argument("--gradcam-count", type=int, default=6)
    parser.add_argument("--limit-folds", type=int, default=0)
    parser.add_argument("--save-fold-models", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parse_config_args(parser, argv)


def main(argv: list[str] | None = None) -> None:
    run_with_args(parse_args(argv))


def run_with_args(args: argparse.Namespace) -> None:
    validate_output_path(Path(args.output_root))
    validate_output_path(
        Path(args.report_dir) if args.report_dir else Path(args.output_root) / "reports"
    )
    dataset_root = Path(args.dataset_root).resolve()
    folds_path = resolve_folds_path(dataset_root, args.folds)
    output_root = Path(args.output_root).resolve()
    report_dir = (Path(args.report_dir) if args.report_dir else output_root / "reports").resolve()
    rows = only_v1_rows(read_manifest(dataset_root / "labels" / "all_labels.csv"))
    has_train_dataset = str(args.train_dataset_root) not in {"", "."}
    train_dataset_root = (
        Path(args.train_dataset_root).resolve() if has_train_dataset else dataset_root
    )
    train_pool_rows = (
        only_v1_rows(read_manifest(train_dataset_root / "labels" / "all_labels.csv"))
        if has_train_dataset
        else None
    )
    needs_target_domain = str(args.domain_adaptation) != "none" or int(args.ssl_pretrain_epochs) > 0
    target_dataset_root = resolve_target_dataset_root(
        args=args,
        dataset_root=dataset_root,
        has_train_dataset=has_train_dataset,
        needs_target_domain=needs_target_domain,
    )
    target_rows = (
        only_v1_rows(read_manifest(target_dataset_root / "labels" / "all_labels.csv"))
        if target_dataset_root is not None
        else None
    )
    folds = read_jsonl(folds_path.resolve())
    k_values = parse_int_list(str(args.k_values))
    seeds = parse_int_list(str(args.seeds))
    validate_fold_plan(folds, rows, k_values=k_values, seeds=seeds)
    if int(args.limit_folds) > 0:
        folds = folds[: int(args.limit_folds)]
    input_files = {
        "dataset_manifest": dataset_root / "labels" / "all_labels.csv",
        "folds": folds_path.resolve(),
        "training_manifest": train_dataset_root / "labels" / "all_labels.csv",
    }
    if target_dataset_root is not None:
        input_files["target_manifest"] = target_dataset_root / "labels" / "all_labels.csv"
    prepare_campaign(output_root, vars(args), inputs=input_files, resume=bool(args.resume))
    device = resolve_device(str(args.device))
    print(f"[INFO] CNN LOOCV device: {device}")

    summary_rows: list[dict[str, object]] = []
    model_name = DEFAULT_MODEL_NAME
    for k in k_values:
        for seed in seeds:
            run_dir = output_root / model_name / f"k{k}_seed{seed}"
            predictions = run_k_seed(
                rows=rows,
                train_pool_rows=train_pool_rows,
                folds=folds,
                dataset_root=dataset_root,
                train_dataset_root=train_dataset_root,
                target_dataset_root=target_dataset_root,
                target_rows=target_rows,
                run_dir=run_dir,
                k=k,
                seed=seed,
                args=args,
                device=device,
            )
            summary_rows.append(
                summarize_predictions(
                    predictions,
                    k=k,
                    seed=seed,
                    run_dir=run_dir,
                    report_dir=report_dir,
                    model_name=model_name,
                )
            )
    write_summary_reports(summary_rows, report_dir)


def only_v1_rows(rows: Sequence[BrugadaImageRow]) -> list[BrugadaImageRow]:
    return [row for row in rows if row.lead.upper() == ONLY_LEAD]


def run_k_seed(
    *,
    rows: Sequence[BrugadaImageRow],
    train_pool_rows: Sequence[BrugadaImageRow] | None,
    folds: Sequence[dict[str, Any]],
    dataset_root: Path,
    train_dataset_root: Path,
    target_dataset_root: Path | None,
    target_rows: Sequence[BrugadaImageRow] | None,
    run_dir: Path,
    k: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    external_train_mode = str(getattr(args, "external_train_mode", "shared"))
    if train_pool_rows is not None and external_train_mode == "shared":
        return run_k_seed_shared_external_training(
            rows=rows,
            train_pool_rows=train_pool_rows,
            folds=folds,
            dataset_root=dataset_root,
            train_dataset_root=train_dataset_root,
            target_dataset_root=target_dataset_root,
            target_rows=target_rows,
            run_dir=run_dir,
            k=k,
            seed=seed,
            args=args,
            device=device,
        )

    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        run_dir / "run_config.json",
        {
            "architecture": DEFAULT_MODEL_NAME,
            "k": k,
            "seed": seed,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "threshold_strategy": str(args.threshold_strategy),
            "loss_pos_weight": str(args.loss_pos_weight),
            "resnet_weights": str(args.resnet_weights),
            "domain_adaptation": str(args.domain_adaptation),
            "domain_adaptation_weight": float(args.domain_adaptation_weight),
            "ssl_pretrain_epochs": int(args.ssl_pretrain_epochs),
            "target_dataset_root": ""
            if target_dataset_root is None
            else target_dataset_root.as_posix(),
            "target_domain_policy": "fold_exclusive" if target_rows is not None else "none",
        },
    )
    output_path = run_dir / "fold_predictions.jsonl"
    resume = bool(getattr(args, "resume", True))
    existing = existing_keys(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()
    predictions: list[dict[str, object]] = load_existing_predictions(output_path)
    gradcam_written = 0
    for fold in folds:
        fold_id = int(fold["fold_id"])
        test_patient_id = str(fold["test_patient_id"])
        key = f"{fold_id}:{test_patient_id}:{k}:{seed}"
        if key in existing:
            continue
        selection = selection_for(fold, k=k, seed=seed)
        context_ids = selection["context_patient_ids"]
        validation_ids = selection["validation_patient_ids"]
        current_train_root = dataset_root
        if train_pool_rows is None:
            train_rows = rows_for_patient_ids(rows, context_ids)
            train_context_ids = context_ids
        else:
            train_context_ids = select_qrs_training_patient_ids(
                train_pool_rows,
                k=k,
                seed=seed + fold_id * 100_003,
            )
            train_rows = rows_for_patient_ids(train_pool_rows, train_context_ids)
            current_train_root = train_dataset_root
        val_rows = rows_for_patient_ids(rows, validation_ids)
        test_rows = rows_for_patient_ids(rows, [test_patient_id])
        print(f"[INFO] CNN k={k} seed={seed} fold={fold_id} test_patient={test_patient_id}")
        result = train_and_predict_fold(
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            train_dataset_root=current_train_root,
            eval_dataset_root=dataset_root,
            target_dataset_root=target_dataset_root,
            target_rows=exclude_patient_rows(target_rows, test_patient_id),
            fold_dir=run_dir / f"fold_{fold_id:04d}",
            fold_id=fold_id,
            seed=seed,
            args=args,
            device=device,
            write_gradcam=gradcam_written < int(args.gradcam_count),
        )
        gradcam_written += int(result.pop("gradcam_written", 0))
        result.update(
            {
                "key": key,
                "fold_id": fold_id,
                "k": k,
                "seed": seed,
                "test_patient_id": test_patient_id,
                "context_patient_ids": "|".join(train_context_ids),
                "validation_patient_ids": "|".join(validation_ids),
            }
        )
        append_jsonl(output_path, result)
        predictions.append(result)
        write_csv(run_dir / "fold_predictions.csv", predictions)
    write_csv(run_dir / "fold_predictions.csv", predictions)
    return predictions


def run_k_seed_shared_external_training(
    *,
    rows: Sequence[BrugadaImageRow],
    train_pool_rows: Sequence[BrugadaImageRow],
    folds: Sequence[dict[str, Any]],
    dataset_root: Path,
    train_dataset_root: Path,
    target_dataset_root: Path | None,
    target_rows: Sequence[BrugadaImageRow] | None,
    run_dir: Path,
    k: int,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "fold_predictions.jsonl"
    resume = bool(getattr(args, "resume", True))
    existing = existing_keys(output_path) if resume else set()
    if output_path.exists() and not resume:
        output_path.unlink()
    predictions: list[dict[str, object]] = load_existing_predictions(output_path)
    train_context_ids = select_qrs_training_patient_ids(
        train_pool_rows,
        k=k,
        seed=seed,
    )
    train_rows = rows_for_patient_ids(train_pool_rows, train_context_ids)
    model, preprocess, training_payload = train_qrs_model(
        train_rows=train_rows,
        train_dataset_root=train_dataset_root,
        target_dataset_root=target_dataset_root,
        target_rows=target_rows,
        run_dir=run_dir / "shared_training",
        seed=seed,
        args=args,
        device=device,
    )
    save_json(
        run_dir / "run_config.json",
        {
            "architecture": DEFAULT_MODEL_NAME,
            "external_train_mode": "shared",
            "train_dataset_root": train_dataset_root.as_posix(),
            "train_context_patient_ids": train_context_ids,
            "k": k,
            "seed": seed,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "threshold_strategy": str(args.threshold_strategy),
            "loss_pos_weight": str(args.loss_pos_weight),
            "resnet_weights": str(args.resnet_weights),
            "domain_adaptation": str(args.domain_adaptation),
            "domain_adaptation_weight": float(args.domain_adaptation_weight),
            "ssl_pretrain_epochs": int(args.ssl_pretrain_epochs),
            "target_dataset_root": ""
            if target_dataset_root is None
            else target_dataset_root.as_posix(),
            "target_domain_policy": "shared_transductive" if target_rows is not None else "none",
        },
    )

    gradcam_written = 0
    for fold in folds:
        fold_id = int(fold["fold_id"])
        test_patient_id = str(fold["test_patient_id"])
        key = f"{fold_id}:{test_patient_id}:{k}:{seed}"
        if key in existing:
            continue
        selection = selection_for(fold, k=k, seed=seed)
        val_rows = rows_for_patient_ids(rows, selection["validation_patient_ids"])
        test_rows = rows_for_patient_ids(rows, [test_patient_id])
        print(
            f"[INFO] CNN shared-train k={k} seed={seed} fold={fold_id} "
            f"test_patient={test_patient_id}"
        )
        result = predict_fold_with_trained_model(
            model=model,
            preprocess=preprocess,
            val_rows=val_rows,
            test_rows=test_rows,
            eval_dataset_root=dataset_root,
            fold_dir=run_dir / f"fold_{fold_id:04d}",
            fold_id=fold_id,
            seed=seed,
            args=args,
            device=device,
            write_gradcam=gradcam_written < int(args.gradcam_count),
        )
        gradcam_written += int(result.pop("gradcam_written", 0))
        result.update(
            {
                **training_payload,
                "key": key,
                "fold_id": fold_id,
                "k": k,
                "seed": seed,
                "test_patient_id": test_patient_id,
                "context_patient_ids": "|".join(train_context_ids),
                "validation_patient_ids": "|".join(selection["validation_patient_ids"]),
            }
        )
        append_jsonl(output_path, result)
        predictions.append(result)
        write_csv(run_dir / "fold_predictions.csv", predictions)
    write_csv(run_dir / "fold_predictions.csv", predictions)
    return predictions


def select_qrs_training_patient_ids(
    rows: Sequence[BrugadaImageRow],
    *,
    k: int,
    seed: int,
) -> list[str]:
    qrs_rows = [row for row in rows if row.has_qrs_labels]
    if not qrs_rows:
        raise ValueError("Training dataset must contain explicit QRS finding labels.")
    patients = patients_from_rows(qrs_rows)
    return select_context_patient_ids(
        patients,
        test_patient_id="__real_eval_patient__",
        k=k,
        seed=seed,
    )


def resolve_target_dataset_root(
    *,
    args: argparse.Namespace,
    dataset_root: Path,
    has_train_dataset: bool,
    needs_target_domain: bool,
) -> Path | None:
    explicit = str(args.target_dataset_root)
    if explicit not in {"", "."}:
        return Path(args.target_dataset_root).resolve()
    if needs_target_domain and has_train_dataset:
        return dataset_root
    if needs_target_domain:
        raise ValueError(
            "--target-dataset-root is required for domain adaptation when no "
            "external train dataset is used."
        )
    return None


_LEGACY_EXPORTS = {
    "build_resnet": "ecg_few.cnn.morphology.models",
    "build_model": "ecg_few.cnn.morphology.models",
    "resolve_device": "ecg_few.cnn.device",
    "set_seed": "ecg_few.cnn.device",
    "make_loader": "ecg_few.cnn.morphology.training",
    "make_target_loader": "ecg_few.cnn.morphology.training",
    "run_train_epoch": "ecg_few.cnn.morphology.training",
    "run_domain_train_epoch": "ecg_few.cnn.morphology.training",
    "evaluate_loss": "ecg_few.cnn.morphology.training",
    "predict_rows": "ecg_few.cnn.morphology.training",
    "loss_pos_weight": "ecg_few.cnn.morphology.training",
    "parse_float_list": "ecg_few.cnn.morphology.training",
    "exclude_patient_rows": "ecg_few.cnn.morphology.training",
    "select_patient_threshold": "ecg_few.cnn.morphology.thresholds",
    "score_patient_threshold": "ecg_few.cnn.morphology.thresholds",
    "patient_reference_labels": "ecg_few.cnn.morphology.thresholds",
    "patient_predictions_at_threshold": "ecg_few.cnn.morphology.thresholds",
    "patient_probability_matrix": "ecg_few.cnn.morphology.thresholds",
    "patient_condition_probabilities": "ecg_few.cnn.morphology.thresholds",
    "lead_probabilities": "ecg_few.cnn.morphology.thresholds",
    "GradCAM": "ecg_few.cnn.morphology.gradcam",
    "save_gradcam_panel": "ecg_few.cnn.morphology.gradcam",
    "gradcam_target_module": "ecg_few.cnn.morphology.gradcam",
    "plot_confusion_matrix": "ecg_few.reporting.plots",
    "plot_curve": "ecg_few.reporting.plots",
    "plot_metric_by_k": "ecg_few.reporting.plots",
    "summarize_predictions": "ecg_few.reporting.cnn",
    "probability_metrics_payload": "ecg_few.reporting.cnn",
    "write_summary_reports": "ecg_few.reporting.cnn",
    "aggregate_by_k": "ecg_few.reporting.cnn",
    "write_confusion_matrices_by_k": "ecg_few.reporting.cnn",
    "train_and_predict_fold": "ecg_few.cnn.morphology.folds",
    "train_qrs_model": "ecg_few.cnn.morphology.folds",
    "predict_fold_with_trained_model": "ecg_few.cnn.morphology.folds",
}


def __getattr__(name: str):
    from importlib import import_module

    if name in _LEGACY_EXPORTS:
        return getattr(import_module(_LEGACY_EXPORTS[name]), name)
    settings = import_module("ecg_few.cnn.morphology.settings")
    if hasattr(settings, name):
        return getattr(settings, name)
    raise AttributeError(name)


if __name__ == "__main__":
    main()
