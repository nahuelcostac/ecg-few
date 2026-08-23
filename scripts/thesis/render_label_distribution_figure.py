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
GRAY = "#666666"
GRID = "#D8D8D8"
NORMAL = "#B7B7B7"
PARTIAL = "#5B8FC9"
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
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.edgecolor": BLACK,
            "axes.linewidth": 0.65,
            "xtick.major.width": 0.65,
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
    labels = [row["label"] for row in rows]
    counts = [int(row["count"]) for row in rows]
    colors = [
        {"normal": NORMAL, "partial": PARTIAL, "target": TARGET}[row["group"]] for row in rows
    ]

    fig, ax = plt.subplots(figsize=(6.45, 2.75))
    positions = range(len(rows))
    bars = ax.barh(positions, counts, height=0.62, color=colors, edgecolor="none")

    ax.set_yticks(list(positions), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Número de pacientes")
    ax.set_xlim(0, 38)
    ax.set_xticks([0, 10, 20, 30])
    ax.grid(axis="x", color=GRID, linewidth=0.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=5, colors=BLACK)
    ax.tick_params(axis="x", length=3, colors=GRAY)

    for bar, count in zip(bars, counts, strict=True):
        ax.text(
            count + 0.65,
            bar.get_y() + bar.get_height() / 2,
            str(count),
            ha="left",
            va="center",
            color=BLACK,
            fontsize=8.2,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout(pad=0.35)

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
