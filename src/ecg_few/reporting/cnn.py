"""Cnn for ECG experiments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from ecg_few.evaluation.metrics import brugada_metrics
from ecg_few.experiment_io import save_json, validate_output_path, write_csv
from ecg_few.reporting.plots import plot_confusion_matrix, plot_curve, plot_metric_by_k


def summarize_predictions(
    predictions: Sequence[dict[str, object]],
    *,
    k: int,
    seed: int,
    run_dir: Path,
    report_dir: Path,
    model_name: str,
) -> dict[str, object]:
    validate_output_path(report_dir)
    y_true = np.asarray([int(row["true_label"]) for row in predictions], dtype=np.int64)
    y_pred = np.asarray([int(row["pred_label"]) for row in predictions], dtype=np.int64)
    y_prob = np.asarray(
        [float(row["prob_derived_brugada"]) for row in predictions], dtype=np.float64
    )
    metrics = brugada_metrics(y_true, y_pred)
    probability_metrics = probability_metrics_payload(
        y_true,
        y_prob,
        report_dir,
        prefix=f"cnn_k{k}_seed{seed}",
    )
    plot_confusion_matrix(run_dir / "confusion_matrix.png", metrics["counts"])
    save_json(run_dir / "metrics.json", {"metrics": metrics, **probability_metrics})
    return {
        "model_family": "cnn",
        "model": model_name,
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
        "roc_auc": probability_metrics["roc_auc"],
        "average_precision": probability_metrics["average_precision"],
        "predictions": (run_dir / "fold_predictions.csv").as_posix(),
        "output_dir": run_dir.as_posix(),
    }


def probability_metrics_payload(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    report_dir: Path,
    *,
    prefix: str,
) -> dict[str, float | None]:
    validate_output_path(report_dir)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return {"roc_auc": None, "average_precision": None}
    roc_auc = float(roc_auc_score(y_true, y_prob))
    average_precision = float(average_precision_score(y_true, y_prob))
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    curves_dir = report_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    plot_curve(
        curves_dir / f"{prefix}_roc.png",
        x=fpr,
        y=tpr,
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"CNN ROC {prefix}",
    )
    plot_curve(
        curves_dir / f"{prefix}_pr.png",
        x=recall,
        y=precision,
        xlabel="Recall",
        ylabel="Precision",
        title=f"CNN PR {prefix}",
    )
    return {"roc_auc": roc_auc, "average_precision": average_precision}


def write_summary_reports(rows: Sequence[dict[str, object]], report_dir: Path) -> None:
    validate_output_path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "cnn_summary_by_seed.csv", rows)
    save_json(report_dir / "cnn_summary_by_seed.json", list(rows))
    by_k = aggregate_by_k(rows)
    write_csv(report_dir / "cnn_summary_by_k.csv", by_k)
    write_confusion_matrices_by_k(report_dir / "confusion_by_k", rows, prefix="cnn")
    plot_metric_by_k(
        report_dir / "balanced_accuracy_by_k.png",
        by_k,
        metric="balanced_accuracy_mean",
        ylabel="Balanced accuracy",
        title="CNN LOOCV balanced accuracy by k",
    )
    plot_metric_by_k(
        report_dir / "f1_by_k.png",
        by_k,
        metric="f1_mean",
        ylabel="F1",
        title="CNN LOOCV F1 by k",
    )
    print(f"[OK] Wrote CNN LOOCV report: {report_dir}")


def aggregate_by_k(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["k"]), []).append(row)
    out: list[dict[str, object]] = []
    for k, group in sorted(grouped.items()):
        payload: dict[str, object] = {"k": k, "n_runs": len(group)}
        for count_name in ("tp", "tn", "fp", "fn"):
            payload[count_name] = sum(int(row.get(count_name, 0) or 0) for row in group)
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "roc_auc",
            "average_precision",
        ):
            values = [
                float(row[metric]) for row in group if row.get(metric) not in {None, "", "None"}
            ]
            payload[f"{metric}_mean"] = float(np.mean(values)) if values else ""
            payload[f"{metric}_std"] = float(np.std(values, ddof=0)) if values else ""
        out.append(payload)
    return out


def write_confusion_matrices_by_k(
    output_dir: Path,
    rows: Sequence[dict[str, object]],
    *,
    prefix: str,
) -> None:
    validate_output_path(output_dir)
    grouped: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(int(row["k"]), []).append(row)
    for k, group in sorted(grouped.items()):
        counts = {
            name: sum(int(row.get(name, 0) or 0) for row in group)
            for name in ("tp", "tn", "fp", "fn")
        }
        plot_confusion_matrix(output_dir / f"{prefix}_k{k}_confusion_matrix.png", counts)
