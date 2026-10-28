#!/usr/bin/env python3
# ruff: noqa: I001
"""Render thesis-ready VLM and CNN/VLM comparison figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOOCV_DIR = ROOT / "outputs" / "final_results_tfg" / "loocv"
DEFAULT_OUTPUT_DIR = ROOT / "thesis" / "thesis" / "assets" / "results"

MODEL = "google/gemma-4-E4B-it"
KEEP_CONDITIONS = ("zero_shot", "estandar", "balanced")

BLUE = "#005BBB"
RED = "#D71920"
GREEN = "#2E7D32"
ORANGE = "#D27D00"
BLACK = "#111111"
GRAY = "#666666"
GRID = "#B8B8B8"
DOMAIN_GAP_YLIMS = {
    "balanced_accuracy": (0.42, 0.96),
    "f1": (0.00, 0.75),
}
DOMAIN_GAP_HUCA_COLORS = {
    BLUE: "#6FA8E8",
    GREEN: "#75B978",
    ORANGE: "#F0A53A",
}

DOMAIN_ADAPTATION_COLORS = {
    "baseline": BLACK,
    "ssl": GRAY,
    "coral": BLUE,
    "mmd": RED,
    "dann": "#7A4AB8",
}

CONDITION_STYLE = {
    "zero_shot": {
        "label": "zero-shot",
        "color": BLACK,
        "linestyle": "None",
        "marker": "s",
    },
    "estandar": {"label": "Gemma 4 estándar", "color": GREEN, "linestyle": "--", "marker": "o"},
    "balanced": {
        "label": "Gemma 4 balanceado",
        "color": ORANGE,
        "linestyle": "-",
        "marker": "^",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render thesis comparison figures.")
    parser.add_argument(
        "--loocv-dir",
        type=Path,
        default=DEFAULT_LOOCV_DIR,
        help="Directory containing the final LOOCV report folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where thesis-ready figures are written.",
    )
    parser.add_argument("--model", default=MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    cnn_sim = read_path_csv(args.loocv_dir / "cnn_simulator_qrs" / "cnn_summary_by_k.csv")
    cnn_real = read_path_csv(args.loocv_dir / "cnn" / "cnn_summary_by_k.csv")
    cnn_domain_adaptation = read_path_csv(
        args.loocv_dir / "cnn_domain_adaptation" / "comparison" / "cnn_domain_adaptation_by_k.csv"
    )
    sim_vlm = selected_vlm_rows(
        read_path_csv(
            args.loocv_dir / "vlm_simulator_qrs" / "vlm_summary_by_model_condition_k.csv"
        ),
        model=args.model,
    )
    real_vlm = selected_vlm_rows(
        read_path_csv(args.loocv_dir / "vlm_real_context" / "vlm_summary_by_model_condition_k.csv"),
        model=args.model,
    )
    huca_sim_context_vlm = selected_vlm_rows(
        read_path_csv(args.loocv_dir / "vlm" / "vlm_summary_by_model_condition_k.csv"),
        model=args.model,
    )

    plot_vlm_metric(
        args.output_dir / "vlm_simulator_qrs_ba_by_k.png",
        sim_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud balanceada (BA)",
        title="Gemma 4 en el simulador QRS/ST",
    )
    plot_vlm_metric(
        args.output_dir / "vlm_simulator_qrs_f1_by_k.png",
        sim_vlm,
        metric="f1",
        ylabel="F1",
        title="Gemma 4 en el simulador QRS/ST",
    )
    plot_vlm_metric(
        args.output_dir / "vlm_huca_v1_ba_by_k.png",
        real_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud balanceada (BA)",
        title="Gemma 4 con ECG del HUCA",
    )
    plot_vlm_metric(
        args.output_dir / "vlm_huca_v1_f1_by_k.png",
        real_vlm,
        metric="f1",
        ylabel="F1",
        title="Gemma 4 con ECG del HUCA",
    )
    plot_vlm_metric(
        args.output_dir / "vlm_huca_synthetic_context_ba_by_k.png",
        huca_sim_context_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud balanceada (BA)",
        title="Contexto del simulador en la muestra del HUCA",
    )
    plot_vlm_metric(
        args.output_dir / "vlm_huca_synthetic_context_f1_by_k.png",
        huca_sim_context_vlm,
        metric="f1",
        ylabel="F1",
        title="Contexto del simulador en la muestra del HUCA",
    )

    plot_comparison_metric(
        args.output_dir / "comparison_simulator_ba_by_k.png",
        cnn_sim,
        sim_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud balanceada (BA)",
        title="Comparación en simulador QRS/ST",
    )
    plot_comparison_metric(
        args.output_dir / "comparison_simulator_f1_by_k.png",
        cnn_sim,
        sim_vlm,
        metric="f1",
        ylabel="F1",
        title="Comparación en simulador QRS/ST",
    )
    plot_comparison_metric(
        args.output_dir / "comparison_huca_ba_by_k.png",
        cnn_real,
        real_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud balanceada (BA)",
        title="Comparación en la muestra del HUCA",
    )
    plot_comparison_metric(
        args.output_dir / "comparison_huca_f1_by_k.png",
        cnn_real,
        real_vlm,
        metric="f1",
        ylabel="F1",
        title="Comparación en la muestra del HUCA",
    )
    plot_huca_sens_spec_best(
        args.output_dir / "comparison_huca_sens_spec_best.png", cnn_real, real_vlm
    )
    plot_cnn_domain_gap_metric(
        args.output_dir / "domain_gap_cnn_ba_by_k.png",
        cnn_sim,
        cnn_real,
        metric="balanced_accuracy",
        ylabel="Exactitud balanceada (BA)",
    )
    plot_cnn_domain_gap_metric(
        args.output_dir / "domain_gap_cnn_f1_by_k.png",
        cnn_sim,
        cnn_real,
        metric="f1",
        ylabel="F1",
    )
    plot_icl_domain_gap_metric(
        args.output_dir / "domain_gap_icl_ba_by_k.png",
        sim_vlm,
        real_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud balanceada (BA)",
    )
    plot_icl_domain_gap_metric(
        args.output_dir / "domain_gap_icl_f1_by_k.png",
        sim_vlm,
        real_vlm,
        metric="f1",
        ylabel="F1",
    )
    plot_icl_context_transfer_metric(
        args.output_dir / "domain_gap_icl_context_ba_by_k.png",
        sim_vlm,
        real_vlm,
        huca_sim_context_vlm,
        metric="balanced_accuracy",
        ylabel="Exactitud balanceada (BA)",
    )
    plot_icl_context_transfer_metric(
        args.output_dir / "domain_gap_icl_context_f1_by_k.png",
        sim_vlm,
        real_vlm,
        huca_sim_context_vlm,
        metric="f1",
        ylabel="F1",
    )
    plot_huca_context_sens_spec(
        args.output_dir / "vlm_huca_context_sens_spec.png",
        real_vlm,
        huca_sim_context_vlm,
    )
    plot_domain_adaptation_metric(
        args.output_dir / "cnn_domain_adaptation_balanced_accuracy_by_k.png",
        cnn_domain_adaptation,
        metric="balanced_accuracy",
        ylabel="Exactitud balanceada (BA)",
    )
    plot_domain_adaptation_metric(
        args.output_dir / "cnn_domain_adaptation_f1_by_k.png",
        cnn_domain_adaptation,
        metric="f1",
        ylabel="F1",
    )
    plot_transfer_controls_summary(
        args.output_dir / "transfer_controls_synthetic_to_real_summary.png",
        cnn_domain_adaptation,
        huca_sim_context_vlm,
    )

    plot_confusion(
        args.output_dir / "vlm_simulator_qrs_balanced_k32_confusion_matrix.png",
        row_for(sim_vlm, "balanced", 32),
        title="Gemma 4 balanceado, K=32",
    )
    plot_confusion(
        args.output_dir / "vlm_huca_v1_balanced_k32_confusion_matrix.png",
        row_for(real_vlm, "balanced", 32),
        title="Gemma 4 balanceado, K=32",
    )
    plot_confusion(
        args.output_dir / "vlm_huca_synthetic_context_balanced_k16_confusion_matrix.png",
        row_for(huca_sim_context_vlm, "balanced", 16),
        title="Contexto del simulador, K=16",
    )


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.edgecolor": BLACK,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "savefig.dpi": 300,
        }
    )


def read_path_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_vlm_rows(rows: list[dict[str, str]], *, model: str) -> list[dict[str, str]]:
    rows = normalize_condition_rows(rows)
    selected = [
        row for row in rows if row.get("model") == model and row.get("condition") in KEEP_CONDITIONS
    ]
    if not selected:
        raise ValueError(f"No VLM rows found for model {model!r}")
    return sorted(
        selected,
        key=lambda row: (int(row["k"]), KEEP_CONDITIONS.index(row["condition"])),
    )


def normalize_condition_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for row in rows:
        copy = dict(row)
        if copy.get("condition") == "normal":
            copy["condition"] = "estandar"
        normalized.append(copy)
    return normalized


def f(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    if value in ("", None):
        return np.nan
    return float(value)


def metric_key(metric: str, stat: str) -> str:
    return f"{metric}_{stat}"


def row_for(rows: list[dict[str, str]], condition: str, k: int) -> dict[str, str]:
    for row in rows:
        if row["condition"] == condition and int(row["k"]) == k:
            return row
    raise ValueError(f"Missing condition={condition} k={k}")


def best_row(rows: list[dict[str, str]]) -> dict[str, str]:
    return max(rows, key=lambda row: f(row, "balanced_accuracy_mean"))


def best_condition_row(rows: list[dict[str, str]], condition: str) -> dict[str, str]:
    return best_row(rows_for_condition(rows, condition))


def rows_for_condition(rows: list[dict[str, str]], condition: str) -> list[dict[str, str]]:
    return sorted(
        [row for row in rows if row["condition"] == condition],
        key=lambda row: int(row["k"]),
    )


def prepare_axis(ax: plt.Axes, ylabel: str, *, include_zero: bool) -> None:
    ax.set_xlabel(r"Número de ejemplos $K$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([0, 2, 4, 8, 16, 32] if include_zero else [2, 4, 8, 16, 32])
    ax.set_xlim((-1.5, 33.5) if include_zero else (0.5, 33.5))
    ax.grid(True, which="major", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def plot_vlm_metric(
    path: Path,
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.95, 2.45))
    for condition in KEEP_CONDITIONS:
        condition_rows = rows_for_condition(rows, condition)
        style = CONDITION_STYLE[condition]
        k_values = [int(row["k"]) for row in condition_rows]
        means = [f(row, metric_key(metric, "mean")) for row in condition_rows]
        stds = [f(row, metric_key(metric, "std")) for row in condition_rows]
        ax.errorbar(
            k_values,
            means,
            yerr=stds,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.2 if condition != "zero_shot" else 3.8,
            linewidth=1.15 if condition != "zero_shot" else 0,
            capsize=2.5,
            elinewidth=0.75,
            label=style["label"],
        )
    ax.set_title(title, pad=3)
    prepare_axis(ax, ylabel, include_zero=True)
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, borderpad=0.25, handlelength=1.6)
    save(fig, path)


def plot_comparison_metric(
    path: Path,
    cnn_rows: list[dict[str, object]],
    vlm_rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4.25, 2.55))
    k_values = [int(row["k"]) for row in cnn_rows]
    ax.errorbar(
        k_values,
        [f(row, metric_key(metric, "mean")) for row in cnn_rows],
        yerr=[f(row, metric_key(metric, "std")) for row in cnn_rows],
        color=BLUE,
        marker="o",
        markersize=3.2,
        linewidth=1.15,
        capsize=2.5,
        elinewidth=0.75,
        label="CNN",
    )
    for condition in ("estandar", "balanced"):
        style = CONDITION_STYLE[condition]
        condition_rows = rows_for_condition(vlm_rows, condition)
        ax.errorbar(
            [int(row["k"]) for row in condition_rows],
            [f(row, metric_key(metric, "mean")) for row in condition_rows],
            yerr=[f(row, metric_key(metric, "std")) for row in condition_rows],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.2,
            linewidth=1.15,
            capsize=2.5,
            elinewidth=0.75,
            label=style["label"],
        )
    ax.set_title(title, pad=3)
    prepare_axis(ax, ylabel, include_zero=False)
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, borderpad=0.25, handlelength=1.6)
    save(fig, path)


def plot_huca_sens_spec_best(
    path: Path,
    cnn_rows: list[dict[str, str]],
    vlm_rows: list[dict[str, str]],
) -> None:
    cnn = best_row(cnn_rows)
    zero_shot = row_for(vlm_rows, "zero_shot", 0)
    estandar = best_condition_row(vlm_rows, "estandar")
    balanced = best_condition_row(vlm_rows, "balanced")
    rows = [
        (
            f"CNN\nK={cnn['k']}",
            f(cnn, "sensitivity_mean"),
            f(cnn, "specificity_mean"),
        ),
        ("Zero-shot", f(zero_shot, "sensitivity_mean"), f(zero_shot, "specificity_mean")),
        (
            f"Gemma 4 estándar\nK={estandar['k']}",
            f(estandar, "sensitivity_mean"),
            f(estandar, "specificity_mean"),
        ),
        (
            f"Gemma 4 balanceado\nK={balanced['k']}",
            f(balanced, "sensitivity_mean"),
            f(balanced, "specificity_mean"),
        ),
    ]
    labels = [row[0] for row in rows]
    sensitivity = [row[1] for row in rows]
    specificity = [row[2] for row in rows]
    x = np.arange(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=(4.6, 2.65))
    ax.bar(x - width / 2, sensitivity, width, label="Sensibilidad", color=RED)
    ax.bar(x + width / 2, specificity, width, label="Especificidad", color=BLUE)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Valor")
    ax.set_xticks(x, labels=labels, rotation=0, ha="center")
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, framealpha=1.0, borderpad=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    save(fig, path)


def plot_cnn_domain_gap_metric(
    path: Path,
    cnn_sim: list[dict[str, str]],
    cnn_real: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.95, 2.45))
    plot_domain_pair(
        ax,
        cnn_sim,
        cnn_real,
        metric=metric,
        label="CNN",
        color=BLUE,
        marker="o",
    )
    prepare_domain_gap_axis(ax, metric=metric, ylabel=ylabel)
    ax.legend(loc="lower right", frameon=True, framealpha=1.0, borderpad=0.25)
    save(fig, path)


def plot_icl_domain_gap_metric(
    path: Path,
    sim_vlm: list[dict[str, str]],
    real_vlm: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3.95, 2.45))
    for condition, marker in (("estandar", "s"), ("balanced", "^")):
        style = CONDITION_STYLE[condition]
        plot_domain_pair(
            ax,
            rows_for_condition(sim_vlm, condition),
            rows_for_condition(real_vlm, condition),
            metric=metric,
            label=style["label"],
            color=style["color"],
            marker=marker,
        )
    prepare_domain_gap_axis(ax, metric=metric, ylabel=ylabel)
    ax.legend(
        loc="lower right",
        fontsize=6,
        frameon=True,
        framealpha=1.0,
        borderpad=0.25,
        handlelength=1.35,
    )
    save(fig, path)


def plot_icl_context_transfer_metric(
    path: Path,
    sim_vlm: list[dict[str, str]],
    real_vlm: list[dict[str, str]],
    huca_sim_context_vlm: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(5.8, 5.15), sharex=True, sharey=True)
    for ax, condition, color, title in (
        (axes[0], "estandar", GREEN, "Gemma 4 estándar"),
        (axes[1], "balanced", ORANGE, "Gemma 4 balanceado"),
    ):
        condition_rows = (
            (rows_for_condition(sim_vlm, condition), "Simulador", "-", color, 1.0),
            (
                rows_for_condition(real_vlm, condition),
                "HUCA, contexto HUCA",
                "--",
                DOMAIN_GAP_HUCA_COLORS[color],
                1.0,
            ),
            (
                rows_for_condition(huca_sim_context_vlm, condition),
                "HUCA, contexto del simulador",
                ":",
                GRAY,
                1.0,
            ),
        )
        for rows, label, linestyle, line_color, alpha in condition_rows:
            ordered = sorted(rows, key=lambda row: int(row["k"]))
            ax.errorbar(
                [int(row["k"]) for row in ordered],
                [f(row, metric_key(metric, "mean")) for row in ordered],
                yerr=optional_yerr(ordered, metric),
                color=line_color,
                linestyle=linestyle,
                marker="o",
                markersize=3.8,
                linewidth=1.2,
                capsize=2.1,
                elinewidth=0.65,
                alpha=alpha,
                label=label,
            )
        ax.set_title(title, pad=3)
        prepare_domain_gap_axis(ax, metric=metric, ylabel=ylabel)
        ax.legend(
            loc="lower right",
            fontsize=8,
            frameon=True,
            framealpha=1.0,
            borderpad=0.25,
            handlelength=1.35,
        )
    axes[0].set_xlabel("")
    save(fig, path)


def plot_huca_context_sens_spec(
    path: Path,
    real_vlm: list[dict[str, str]],
    huca_sim_context_vlm: list[dict[str, str]],
) -> None:
    rows = [
        ("HUCA estándar K=32", row_for(real_vlm, "estandar", 32)),
        ("HUCA balanceado K=32", row_for(real_vlm, "balanced", 32)),
        ("Sintético estándar K=32", row_for(huca_sim_context_vlm, "estandar", 32)),
        ("Sintético balanceado K=16", row_for(huca_sim_context_vlm, "balanced", 16)),
    ]
    labels = [label for label, _ in rows]
    sensitivity = [f(row, "sensitivity_mean") for _, row in rows]
    specificity = [f(row, "specificity_mean") for _, row in rows]
    x = np.arange(len(rows))
    width = 0.34

    fig, ax = plt.subplots(figsize=(4.65, 2.7))
    ax.bar(x - width / 2, sensitivity, width, label="Sensibilidad", color=RED)
    ax.bar(x + width / 2, specificity, width, label="Especificidad", color=BLUE)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Valor")
    ax.set_xticks(x, labels=labels, rotation=18, ha="right")
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=True, framealpha=1.0, borderpad=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    save(fig, path)


def plot_domain_adaptation_metric(
    path: Path,
    domain_rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(4.35, 2.55))
    method_order = ("baseline", "ssl", "coral", "mmd", "dann")
    method_labels = {
        "baseline": "Base",
        "ssl": "SSL",
        "coral": "CORAL",
        "mmd": "MMD",
        "dann": "DANN",
    }
    for method in method_order:
        method_rows = sorted(
            [row for row in domain_rows if row["method"] == method],
            key=lambda row: int(row["k"]),
        )
        ax.errorbar(
            [int(row["k"]) for row in method_rows],
            [f(row, metric_key(metric, "mean")) for row in method_rows],
            yerr=optional_yerr(method_rows, metric),
            color=DOMAIN_ADAPTATION_COLORS[method],
            marker="o",
            markersize=3.0,
            linewidth=1.0,
            capsize=2.0,
            elinewidth=0.65,
            label=method_labels[method],
        )
    prepare_axis(ax, ylabel, include_zero=False)
    ax.set_xticks([16, 32])
    ax.set_xlim(13.5, 34.5)
    ax.legend(
        loc="lower right",
        ncol=2,
        frameon=True,
        framealpha=1.0,
        borderpad=0.25,
        columnspacing=0.8,
        handlelength=1.35,
    )
    save(fig, path)


def plot_transfer_controls_summary(
    path: Path,
    domain_rows: list[dict[str, str]],
    huca_sim_context_vlm: list[dict[str, str]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.45), sharey="row")
    plot_domain_adaptation_on_axis(
        axes[0, 0],
        domain_rows,
        metric="balanced_accuracy",
        ylabel="Exactitud balanceada (BA)",
        title="CNN con adaptación de dominio",
    )
    plot_vlm_on_axis(
        axes[0, 1],
        huca_sim_context_vlm,
        metric="balanced_accuracy",
        ylabel="",
        title="Gemma 4: ejemplos del simulador",
    )
    plot_domain_adaptation_on_axis(
        axes[1, 0],
        domain_rows,
        metric="f1",
        ylabel="F1",
        title="CNN con adaptación de dominio",
    )
    plot_vlm_on_axis(
        axes[1, 1],
        huca_sim_context_vlm,
        metric="f1",
        ylabel="",
        title="Gemma 4: ejemplos del simulador",
    )
    for ax in axes[0]:
        ax.set_ylim(0.45, 0.58)
    for ax in axes[1]:
        ax.set_ylim(0.0, 0.55)
    save(fig, path)


def plot_domain_adaptation_on_axis(
    ax: plt.Axes,
    domain_rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    method_order = ("baseline", "ssl", "coral", "mmd", "dann")
    method_labels = {
        "baseline": "Base",
        "ssl": "SSL",
        "coral": "CORAL",
        "mmd": "MMD",
        "dann": "DANN",
    }
    for method in method_order:
        method_rows = sorted(
            [row for row in domain_rows if row["method"] == method],
            key=lambda row: int(row["k"]),
        )
        ax.errorbar(
            [int(row["k"]) for row in method_rows],
            [f(row, metric_key(metric, "mean")) for row in method_rows],
            yerr=optional_yerr(method_rows, metric),
            color=DOMAIN_ADAPTATION_COLORS[method],
            marker="o",
            markersize=3.0,
            linewidth=1.0,
            capsize=2.0,
            elinewidth=0.65,
            label=method_labels[method],
        )
    ax.set_title(title, pad=3)
    ax.set_xlabel(r"Número de ejemplos $K$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([16, 32])
    ax.set_xlim(13.5, 34.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(
        loc="lower right",
        ncol=2,
        fontsize=5.8,
        frameon=True,
        framealpha=1.0,
        borderpad=0.25,
        columnspacing=0.8,
        handlelength=1.25,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def plot_vlm_on_axis(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> None:
    for condition in KEEP_CONDITIONS:
        condition_rows = rows_for_condition(rows, condition)
        style = CONDITION_STYLE[condition]
        ax.errorbar(
            [int(row["k"]) for row in condition_rows],
            [f(row, metric_key(metric, "mean")) for row in condition_rows],
            yerr=optional_yerr(condition_rows, metric),
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=3.0,
            linewidth=1.0 if condition != "zero_shot" else 0,
            capsize=2.0,
            elinewidth=0.65,
            label=style["label"],
        )
    ax.set_title(title, pad=3)
    ax.set_xlabel(r"Número de ejemplos $K$")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 2, 4, 8, 16, 32])
    ax.set_xlim(-1.5, 33.5)
    ax.grid(True, which="major", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left" if metric == "f1" else "lower right",
        fontsize=5.8,
        frameon=True,
        framealpha=1.0,
        borderpad=0.25,
        handlelength=1.25,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def optional_yerr(rows: list[dict[str, object]], metric: str) -> list[float] | None:
    values = [f(row, metric_key(metric, "std")) for row in rows]
    if all(np.isnan(value) for value in values):
        return None
    return values


def prepare_domain_gap_axis(ax: plt.Axes, *, metric: str, ylabel: str) -> None:
    ax.set_xlabel(r"Número de ejemplos $K$")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*DOMAIN_GAP_YLIMS[metric])
    ax.set_xlim(0.5, 33.5)
    ax.set_xticks([2, 4, 8, 16, 32])
    ax.grid(True, which="major", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)


def plot_domain_pair(
    ax: plt.Axes,
    sim_rows: list[dict[str, object]],
    real_rows: list[dict[str, object]],
    *,
    metric: str,
    label: str,
    color: str,
    marker: str,
) -> None:
    for rows, domain_label, linestyle, domain_color, alpha in (
        (sim_rows, "simulador", "-", color, 1.0),
        (real_rows, "muestra del HUCA", "--", DOMAIN_GAP_HUCA_COLORS[color], 1.0),
    ):
        ordered = sorted(rows, key=lambda row: int(row["k"]))
        ax.errorbar(
            [int(row["k"]) for row in ordered],
            [f(row, metric_key(metric, "mean")) for row in ordered],
            yerr=[f(row, metric_key(metric, "std")) for row in ordered],
            color=domain_color,
            linestyle=linestyle,
            marker=marker,
            markersize=3.0,
            linewidth=1.05,
            capsize=2.2,
            elinewidth=0.65,
            alpha=alpha,
            label=f"{label}, {domain_label}",
        )


def plot_confusion(path: Path, row: dict[str, str], *, title: str) -> None:
    matrix = np.array(
        [
            [int(row["tn"]), int(row["fp"])],
            [int(row["fn"]), int(row["tp"])],
        ]
    )
    fig, ax = plt.subplots(figsize=(2.55, 2.35))
    ax.pcolormesh(
        np.arange(3) - 0.5,
        np.arange(3) - 0.5,
        matrix,
        cmap="Blues",
        vmin=0,
        shading="flat",
        edgecolors="face",
        linewidth=0.2,
    )
    ax.set_ylim(1.5, -0.5)
    ax.set_aspect("equal")
    ax.set_title(title, pad=4)
    ax.set_xticks([0, 1], labels=["Pred.\nnegativa", "Pred.\npositiva"])
    ax.set_yticks([0, 1], labels=["Ref.\nnegativa", "Ref.\npositiva"])
    threshold = matrix.max() * 0.5
    for i in range(2):
        for j in range(2):
            color = "white" if matrix[i, j] > threshold else BLACK
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
    save(fig, path)


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.35)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
