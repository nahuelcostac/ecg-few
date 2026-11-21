"""Clinical for ECG experiments."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ecg_few.evaluation.metrics import brugada_metrics
from ecg_few.experiment_io import validate_output_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecg_few.cnn.clinical.settings import CONDITIONS
from ecg_few.experiment_io import read_csv, save_json, write_csv
from ecg_few.reporting.plots import plot_confusion_matrix


def summarize_predictions(
    predictions: Sequence[dict[str, object]],
    *,
    k: int,
    seed: int,
    condition: str,
    run_dir: Path,
    report_dir: Path,
) -> dict[str, object]:
    validate_output_path(report_dir)
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
    validate_output_path(report_dir)
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
        title=f"CNN clÃ­nica, {condition}, K={k}, semilla={seed}",
    )
    save_curve(
        curves_dir / f"cnn_clinical_{condition}_k{k}_seed{seed}_pr.png",
        y_true,
        y_prob,
        curve="pr",
        title=f"CNN clÃ­nica, {condition}, K={k}, semilla={seed}",
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
        ax.set_ylabel("PrecisiÃ³n")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linewidth=0.45, alpha=0.6)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def write_summary_reports(rows: Sequence[dict[str, object]], report_dir: Path) -> None:
    validate_output_path(report_dir)
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
    validate_output_path(output_dir)
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
    labels = {"estandar": "CNN clÃ­nica estÃ¡ndar", "balanced": "CNN clÃ­nica balanceada"}
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
    ax.set_xlabel(r"NÃºmero de ejemplos $K$")
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
        ("estandar", "sensitivity"): "CNN estÃ¡ndar, sensibilidad",
        ("estandar", "specificity"): "CNN estÃ¡ndar, especificidad",
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
    ax.set_xlabel(r"NÃºmero de ejemplos $K$")
    ax.set_ylabel("ProporciÃ³n")
    ax.set_ylim(0, 1.02)
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.grid(True, linewidth=0.45, alpha=0.7)
    ax.legend(fontsize=6.5, frameon=True, ncol=2)
    fig.tight_layout()
    for suffix, dpi in ((".pdf", 300), (".png", 300)):
        fig.savefig(base_path.with_suffix(suffix), dpi=dpi)
    plt.close(fig)
