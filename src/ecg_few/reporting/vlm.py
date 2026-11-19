"""Vlm for ECG experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from ecg_few.evaluation.metrics import brugada_metrics
from ecg_few.experiment_io import validate_output_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecg_few.experiment_io import save_json, write_csv
from ecg_few.vlm.selection import model_slug
from ecg_few.vlm.settings import MORPHOLOGY_TASK


def summarize_predictions(
    predictions: list[dict[str, object]],
    *,
    k: int,
    seed: int,
    task: str,
    condition: str,
    model: str,
    run_dir: Path,
) -> dict[str, object]:
    y_true = [int(row["true_label"]) for row in predictions]
    y_pred = [int(row["pred_label"]) for row in predictions]
    metrics = brugada_metrics(y_true, y_pred)
    save_json(run_dir / "metrics.json", {"metrics": metrics})
    invalid_leads = sum(int(row.get("invalid_leads", 0) or 0) for row in predictions)
    total_leads = sum(
        int(row.get("valid_leads", 0) or 0) + int(row.get("invalid_leads", 0) or 0)
        for row in predictions
    )
    latencies = [float(row.get("latency_seconds", 0.0) or 0.0) for row in predictions]
    return {
        "model_family": "vlm",
        "model": model,
        "model_slug": model_slug(model),
        "task": task,
        "clinical_lead": first_nonempty(predictions, "clinical_lead"),
        "clinical_leads": first_nonempty(predictions, "clinical_leads"),
        "clinical_aggregation": first_nonempty(predictions, "clinical_aggregation"),
        "condition": condition,
        "k": k,
        "seed": seed,
        "n_patients": len(predictions),
        "n_leads": total_leads,
        "json_invalid_leads": invalid_leads,
        "json_invalid_rate": invalid_leads / total_leads if total_leads else 0.0,
        "latency_seconds_mean": float(np.mean(latencies)) if latencies else "",
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
        "roc_auc": "",
        "average_precision": "",
        "predictions": (run_dir / "fold_predictions.csv").as_posix(),
        "output_dir": run_dir.as_posix(),
    }


def first_nonempty(rows: list[dict[str, object]], key: str) -> object:
    for row in rows:
        value = row.get(key)
        if value not in {None, "", "None"}:
            return value
    return ""


def write_summary_reports(rows: list[dict[str, object]], report_dir: Path) -> None:
    validate_output_path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "vlm_summary_by_seed.csv", rows)
    save_json(report_dir / "vlm_summary_by_seed.json", rows)
    by_model_condition_k = aggregate_by_group(
        rows,
        group_keys=("model", "task", "condition", "k"),
    )
    write_csv(report_dir / "vlm_summary_by_k.csv", by_model_condition_k)
    write_csv(report_dir / "vlm_summary_by_model_condition_k.csv", by_model_condition_k)
    write_confusion_matrices_by_group(
        report_dir / "confusion_by_model_condition_k",
        rows,
        prefix="vlm",
    )
    plot_metric_by_group(
        report_dir / "balanced_accuracy_by_k.png",
        by_model_condition_k,
        metric="balanced_accuracy",
    )
    plot_metric_by_group(report_dir / "f1_by_k.png", by_model_condition_k, metric="f1")
    write_campaign_manifest(report_dir / "vlm_campaign_manifest.json", rows)
    print(f"[OK] Wrote VLM LOOCV report: {report_dir}")


def aggregate_by_group(
    rows: list[dict[str, object]],
    *,
    group_keys: tuple[str, ...],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, object]] = []
    sorted_groups = sorted(
        grouped.items(),
        key=lambda item: tuple(str(part) for part in item[0]),
    )
    for key, group in sorted_groups:
        payload: dict[str, object] = {
            group_key: value for group_key, value in zip(group_keys, key, strict=True)
        }
        payload["n_runs"] = len(group)
        for count_name in ("tp", "tn", "fp", "fn", "json_invalid_leads", "n_leads"):
            payload[count_name] = sum(int(row.get(count_name, 0) or 0) for row in group)
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "json_invalid_rate",
            "latency_seconds_mean",
        ):
            values = [
                float(row[metric]) for row in group if row.get(metric) not in {None, "", "None"}
            ]
            payload[f"{metric}_mean"] = float(np.mean(values)) if values else ""
            payload[f"{metric}_std"] = float(np.std(values, ddof=0)) if values else ""
        out.append(payload)
    return out


def write_confusion_matrices_by_group(
    output_dir: Path,
    rows: list[dict[str, object]],
    *,
    prefix: str,
) -> None:
    validate_output_path(output_dir)
    grouped: dict[tuple[str, str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["model_slug"]),
            str(row.get("task", MORPHOLOGY_TASK)),
            str(row["condition"]),
            int(row["k"]),
        )
        grouped.setdefault(key, []).append(row)
    for (slug, task, condition, k), group in sorted(grouped.items()):
        counts = {
            name: sum(int(row.get(name, 0) or 0) for row in group)
            for name in ("tp", "tn", "fp", "fn")
        }
        filename = f"{prefix}_{slug}_{task}_{condition}_k{k}_confusion_matrix.png"
        plot_confusion_matrix(output_dir / filename, counts)


def plot_confusion_matrix(path: Path, counts: dict[str, int]) -> None:
    matrix = np.asarray([[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["pred 0", "pred 1"])
    ax.set_yticks([0, 1], labels=["true 0", "true 1"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_metric_by_group(path: Path, rows: list[dict[str, object]], *, metric: str) -> None:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for row in rows:
        value = row.get(f"{metric}_mean")
        if value in {None, "", "None"}:
            continue
        key = (
            str(row.get("model", "")),
            str(row.get("task", MORPHOLOGY_TASK)),
            str(row.get("condition", "")),
        )
        grouped.setdefault(key, []).append((int(row["k"]), float(value)))
    if not grouped:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (model, task, condition), points in sorted(grouped.items()):
        points = sorted(points)
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=f"{model} / {task} / {condition}",
        )
    ax.set_xlabel("k patients")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"VLM LOOCV {metric.replace('_', ' ')} by k")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_campaign_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    models = sorted({str(row.get("model", "")) for row in rows if row.get("model")})
    conditions = sorted({str(row.get("condition", "")) for row in rows if row.get("condition")})
    tasks = sorted({str(row.get("task", "")) for row in rows if row.get("task")})
    k_values = sorted({int(row["k"]) for row in rows}) if rows else []
    seeds = sorted({int(row["seed"]) for row in rows}) if rows else []
    clinical_leads = sorted(
        {str(row.get("clinical_leads", "")) for row in rows if row.get("clinical_leads")}
    )
    clinical_aggregations = sorted(
        {
            str(row.get("clinical_aggregation", ""))
            for row in rows
            if row.get("clinical_aggregation")
        }
    )
    payload = {
        "models": models,
        "conditions": conditions,
        "tasks": tasks,
        "k_values": k_values,
        "seeds": seeds,
        "clinical_leads": clinical_leads,
        "clinical_aggregations": clinical_aggregations,
        "runs": len(rows),
        "artifacts": {
            "summary_by_seed": "vlm_summary_by_seed.csv",
            "summary_by_model_condition_k": "vlm_summary_by_model_condition_k.csv",
            "confusion_matrices": "confusion_by_model_condition_k/",
            "balanced_accuracy_plot": "balanced_accuracy_by_k.png",
            "f1_plot": "f1_by_k.png",
            "per_run_predictions": (
                "outputs/work/vlm_loocv/<model>/<condition>/k<k>_seed<seed>/fold_predictions.csv"
            ),
        },
    }
    save_json(path, payload)
