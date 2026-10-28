#!/usr/bin/env python3
"""Render the synthetic label-combination distribution used in the thesis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "thesis" / "thesis" / "assets" / "data" / "synthetic_label_combinations.csv"
DEFAULT_OUTPUT_DIR = ROOT / "thesis" / "thesis" / "assets" / "results"

BLACK = "#111111"
GRID = "#B8B8B8"
NORMAL = "#B7B7B7"
PARTIAL = "#005BBB"
TARGET = "#D27D00"


def main() -> None:
    args = parse_args()
    set_style()
    rows = read_counts(args.data)
    validate_counts(rows)
    plot(rows, args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the thesis figure for synthetic label combinations."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="CSV with columns label, count and group.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PDF and PNG figures are written.",
    )
    return parser.parse_args()


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


def read_counts(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_counts(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("The label-combination table is empty.")
    required = {"label", "count", "group"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Expected CSV columns: {sorted(required)}")
    total = sum(int(row["count"]) for row in rows)
    if total != 100:
        raise ValueError(f"Expected 100 synthetic patients, found {total}.")


def plot(rows: list[dict[str, str]], output_dir: Path) -> None:
    labels = [
        row["label"]
        .replace("Inversión de T", "T invertida")
        .replace("inversión de T", "T invertida")
        .replace("elevación del ST", "ST elevado")
        .replace(" + ", "\n+ ")
        for row in rows
    ]
    counts = [int(row["count"]) for row in rows]
    colors = [
        {"normal": NORMAL, "partial": PARTIAL, "target": TARGET}[row["group"]] for row in rows
    ]

    fig, ax = plt.subplots(figsize=(6.45, 3.5))
    positions = range(len(rows))
    bars = ax.bar(positions, counts, width=0.62, color=colors, edgecolor="none")

    ax.set_title("Combinaciones de etiquetas en el simulador", pad=3)
    ax.set_xticks(list(positions), labels)
    ax.set_xlabel("Combinación de etiquetas")
    ax.set_ylabel("Número de pacientes")
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.grid(True, which="major", color=GRID, linewidth=0.45, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=BLACK)

    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.65,
            str(count),
            ha="center",
            va="bottom",
            color=BLACK,
            fontsize=8.2,
        )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
    fig.tight_layout(pad=0.4)

    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"fig_label_distribution.{suffix}",
            bbox_inches="tight",
            pad_inches=0.02,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
