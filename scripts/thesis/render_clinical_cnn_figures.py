#!/usr/bin/env python3
"""Render the canonical direct-clinical-CNN thesis figures and tables."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANONICAL_ROOT = ROOT / "outputs" / "final_results_tfg" / "loocv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "final_results_tfg" / "tfg_figures" / "results"
DEFAULT_THESIS_FIGURES_DIR = ROOT / "thesis" / "thesis" / "assets" / "results"
DEFAULT_THESIS_GENERATED_DIR = ROOT / "thesis" / "thesis" / "generated"
BLUE = "#005BBB"
RED = "#D71920"
GREEN = "#2E7D32"
ORANGE = "#D27D00"
BLACK = "#111111"
GRID = "#B8B8B8"
MODEL = "google/gemma-4-E4B-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render canonical clinical comparison figures.")
    parser.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--thesis-figures-dir", type=Path, default=DEFAULT_THESIS_FIGURES_DIR)
    parser.add_argument("--thesis-generated-dir", type=Path, default=DEFAULT_THESIS_GENERATED_DIR)
    parser.add_argument("--model", default=MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_style()
    cnn_rows = {
        condition: read_csv(
            args.canonical_root / "cnn_clinical" / condition / "cnn_clinical_summary_by_k.csv"
        )
        for condition in ("estandar", "balanced")
    }
    vlm_rows = [
        row
        for row in read_csv(args.canonical_root / "vlm_real_context" / "vlm_summary_by_k.csv")
        if row.get("model") in (None, "", args.model)
        and row.get("condition") in ("estandar", "balanced")
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.thesis_figures_dir.mkdir(parents=True, exist_ok=True)
    figure_bases = (
        plot_performance(
            args.output_dir / "clinical_cnn_vs_gemma_performance",
            cnn_rows,
            vlm_rows,
        ),
        plot_confusions(
            args.output_dir / "clinical_cnn_vs_gemma_confusions",
            best_condition_row(cnn_rows["balanced"], "balanced"),
            best_condition_row(vlm_rows, "balanced"),
        ),
    )
    for base_path in figure_bases:
        for extension in (".pdf", ".png"):
            shutil.copy2(
                base_path.with_suffix(extension),
                args.thesis_figures_dir / base_path.with_suffix(extension).name,
            )
    write_latex_tables(args.thesis_generated_dir, cnn_rows, vlm_rows)
    write_text(
        args.output_dir / "clinical_cnn_manifest.txt",
        "Generated exclusively from outputs/final_results_tfg/loocv aggregate CSV files.\n",
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
            "savefig.dpi": 300,
        }
    )


def rows_by_k(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return sorted(rows, key=lambda row: int(row["k"]))


def plot_performance(
    base_path: Path,
    cnn_rows: dict[str, list[dict[str, str]]],
    vlm_rows: list[dict[str, str]],
) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.35), sharex=True)
    styles = {
        "cnn_estandar": (BLUE, "CNN (HUCA), estándar", "--", "o"),
        "cnn_balanced": (RED, "CNN (HUCA), balanceada", "-", "^"),
        "vlm_estandar": (GREEN, "Gemma 4, estándar", "--", "s"),
        "vlm_balanced": (ORANGE, "Gemma 4, balanceado", "-", "D"),
    }
    for axis, metric, ylabel, ylim in (
        (axes[0], "balanced_accuracy", "Exactitud balanceada (BA)", (0.44, 0.80)),
        (axes[1], "f1", "F1", (0.10, 0.75)),
    ):
        for condition, rows in cnn_rows.items():
            plot_metric_line(axis, rows_by_k(rows), metric, styles[f"cnn_{condition}"])
        for condition in ("estandar", "balanced"):
            selected = rows_by_k([row for row in vlm_rows if row.get("condition") == condition])
            plot_metric_line(axis, selected, metric, styles[f"vlm_{condition}"])
        axis.set_ylabel(ylabel)
        axis.set_ylim(*ylim)
        axis.set_xticks([2, 4, 8, 16, 32])
        axis.grid(True, color=GRID, linewidth=0.45, alpha=0.8)
        axis.set_axisbelow(True)
    axes[0].axhline(0.5, color=BLACK, linewidth=0.7, linestyle=":", alpha=0.85)
    axes[0].text(2.2, 0.505, "azar", fontsize=7, color=BLACK, va="bottom")
    axes[0].set_title("(a) Rendimiento equilibrado")
    axes[1].set_title("(b) Rendimiento de la clase positiva")
    axes[1].set_xlabel(r"Número de ejemplos $K$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=True,
        framealpha=1.0,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=1.25)
    save_figure(fig, base_path)
    return base_path


def plot_metric_line(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    metric: str,
    style: tuple[str, str, str, str],
) -> None:
    color, label, linestyle, marker = style
    axis.errorbar(
        [int(row["k"]) for row in rows],
        [float(row[f"{metric}_mean"]) for row in rows],
        yerr=[float(row[f"{metric}_std"]) for row in rows],
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=3.4,
        linewidth=1.05,
        capsize=2.2,
        label=label,
    )


def plot_confusions(
    base_path: Path,
    cnn_row: dict[str, str],
    vlm_row: dict[str, str],
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.85), constrained_layout=True)
    image = None
    panels = (
        (axes[0], cnn_row, "CNN (HUCA), balanceada"),
        (axes[1], vlm_row, "Gemma 4, balanceado"),
    )
    for axis, row, label in panels:
        counts = np.asarray(
            [[int(row["tn"]), int(row["fp"])], [int(row["fn"]), int(row["tp"])]],
            dtype=int,
        )
        proportions = counts / counts.sum(axis=1, keepdims=True)
        image = axis.pcolormesh(
            np.arange(3) - 0.5,
            np.arange(3) - 0.5,
            proportions,
            cmap="Blues",
            vmin=0,
            vmax=1,
            shading="flat",
            edgecolors="face",
            linewidth=0.2,
        )
        axis.set_ylim(1.5, -0.5)
        axis.set_aspect("equal")
        sensitivity = decimal(float(row["sensitivity_mean"]))
        specificity = decimal(float(row["specificity_mean"]))
        axis.set_title(
            f"{label}, K={int(row['k'])}\nSens. {sensitivity}, esp. {specificity}",
            linespacing=1.25,
        )
        axis.set_xticks([0, 1], labels=["Pred. normal", "Pred. Brugada"])
        axis.set_yticks([0, 1], labels=["Real normal", "Real Brugada"])
        axis.tick_params(axis="x", labelrotation=12)
        for row_index in range(2):
            for column_index in range(2):
                value = proportions[row_index, column_index]
                color = "white" if value >= 0.56 else BLACK
                axis.text(
                    column_index,
                    row_index,
                    f"{counts[row_index, column_index]}\n{100 * value:.1f}%".replace(".", ","),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.035)
        if colorbar.solids is not None:
            colorbar.solids.set_rasterized(False)
            colorbar.solids.set_edgecolor("face")
        colorbar.set_label("Proporción dentro de la clase real", fontsize=8)
    save_figure(fig, base_path)
    return base_path


def best_condition_row(rows: list[dict[str, str]], condition: str) -> dict[str, str]:
    candidates = [row for row in rows if row.get("condition") == condition]
    if not candidates:
        raise ValueError(f"No rows found for condition={condition}.")
    return max(candidates, key=lambda row: float(row["balanced_accuracy_mean"]))


def write_latex_tables(
    generated_dir: Path,
    cnn_rows: dict[str, list[dict[str, str]]],
    vlm_rows: list[dict[str, str]],
) -> None:
    generated_dir.mkdir(parents=True, exist_ok=True)
    full_lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabular}{lrrrrr}",
        r"    \toprule",
        r"    Condición & \(K\) & BA & F1 & Sens. & Esp. \\",
        r"    \midrule",
    ]
    for condition in ("estandar", "balanced"):
        for row in rows_by_k(cnn_rows[condition]):
            full_lines.append(table_row(condition_label(condition), row))
    full_lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \caption{Resultados completos de la CNN entrenada directamente con la "
            r"etiqueta de Brugada del HUCA.}",
            r"  \label{tab:cnn_clinica_resultados}",
            r"  \resultstablenote",
            r"\end{table}",
            "",
        ]
    )
    write_text(generated_dir / "clinical_cnn_results_table.tex", "\n".join(full_lines))

    comparison_rows = (
        ("CNN estándar", best_condition_row(cnn_rows["estandar"], "estandar")),
        ("CNN balanceada", best_condition_row(cnn_rows["balanced"], "balanced")),
        ("Gemma 4 estándar", best_condition_row(vlm_rows, "estandar")),
        ("Gemma 4 balanceado", best_condition_row(vlm_rows, "balanced")),
    )
    comparison_lines = [
        r"\begin{table}[H]",
        r"  \centering",
        r"  \small",
        r"  \begin{tabularx}{\textwidth}{Xrrrrr}",
        r"    \toprule",
        r"    Método & \(K\) & BA & F1 & Sens. & Esp. \\",
        r"    \midrule",
    ]
    comparison_lines.extend(table_row(label, row) for label, row in comparison_rows)
    comparison_lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabularx}",
            r"  \caption{Mejores configuraciones de la CNN entrenada con el HUCA y de Gemma~4 con "
            r"demostraciones de ese mismo conjunto.}",
            r"  \label{tab:comparacion_clinica_cnn}",
            r"  \resultstablenote",
            r"\end{table}",
            "",
        ]
    )
    write_text(
        generated_dir / "clinical_cnn_comparison_table.tex",
        "\n".join(comparison_lines),
    )


def table_row(label: str, row: dict[str, str]) -> str:
    return (
        "    "
        + " & ".join(
            [
                label,
                row["k"],
                format_value(row["balanced_accuracy_mean"], row["balanced_accuracy_std"]),
                format_value(row["f1_mean"], row["f1_std"]),
                format_value(row["sensitivity_mean"], row["sensitivity_std"]),
                format_value(row["specificity_mean"], row["specificity_std"]),
            ]
        )
        + r" \\"
    )


def condition_label(condition: str) -> str:
    return "Estándar" if condition == "estandar" else "Balanceada"


def decimal(value: float) -> str:
    return f"{value:.3f}".replace(".", ",")


def format_value(mean: str, std: str) -> str:
    return r"\(" + decimal(float(mean)) + r" \pm " + decimal(float(std)) + r"\)"


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")


if __name__ == "__main__":
    main()
