#!/usr/bin/env python3
"""Render the patient-level split and K-example selection diagram."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "thesis" / "thesis" / "assets" / "results"

INK = "#1F2328"
MUTED = "#5F6368"
BLUE_FILL = "#E5EEF9"
BLUE_EDGE = "#6E8EB8"
RED_FILL = "#F9E9E7"
RED_EDGE = "#B76E67"
ORANGE_FILL = "#FFF1D8"
ORANGE_EDGE = "#C58A28"
GREEN_FILL = "#E8F3E5"
GREEN_EDGE = "#6D9C64"
PURPLE_FILL = "#F0E8F6"
PURPLE_EDGE = "#8B69A7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the K-example split figure.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG and PDF figures are written.",
    )
    return parser.parse_args()


def add_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    fill: str,
    edge: str,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.25,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=8.8,
        weight="bold",
        color=INK,
    )
    ax.text(
        x + width / 2,
        y + height * 0.29,
        body,
        ha="center",
        va="center",
        fontsize=7.4,
        color=MUTED,
        linespacing=1.15,
    )


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.25,
            color=MUTED,
            connectionstyle=connectionstyle,
        )
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(9.4, 3.45))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    add_box(
        ax,
        x=0.035,
        y=0.65,
        width=0.19,
        height=0.22,
        title="Paciente i\nreservado",
        body="consulta o prueba\nno interviene en la selección",
        fill=RED_FILL,
        edge=RED_EDGE,
    )
    add_box(
        ax,
        x=0.035,
        y=0.17,
        width=0.19,
        height=0.25,
        title="Banco de\nejemplos",
        body="pacientes disponibles\nsin el paciente i",
        fill=BLUE_FILL,
        edge=BLUE_EDGE,
    )
    add_box(
        ax,
        x=0.295,
        y=0.17,
        width=0.19,
        height=0.25,
        title="Selección de\n" + r"$K$ ejemplos",
        body=r"$K \in \{2,4,8,16,32\}$" + "\ntres semillas",
        fill=ORANGE_FILL,
        edge=ORANGE_EDGE,
    )
    add_box(
        ax,
        x=0.555,
        y=0.35,
        width=0.19,
        height=0.22,
        title="CNN",
        body="los ejemplos actualizan\nlos pesos",
        fill=GREEN_FILL,
        edge=GREEN_EDGE,
    )
    add_box(
        ax,
        x=0.555,
        y=0.08,
        width=0.19,
        height=0.22,
        title="VLM/ICL",
        body="los ejemplos forman\nel contexto",
        fill=PURPLE_FILL,
        edge=PURPLE_EDGE,
    )
    add_box(
        ax,
        x=0.815,
        y=0.65,
        width=0.15,
        height=0.22,
        title="Evaluación",
        body="predicción sobre\nel paciente i",
        fill="#F3F3F3",
        edge="#777777",
    )

    add_arrow(ax, (0.225, 0.295), (0.295, 0.295))
    add_arrow(ax, (0.485, 0.31), (0.555, 0.46), connectionstyle="arc3,rad=-0.08")
    add_arrow(ax, (0.485, 0.27), (0.555, 0.19), connectionstyle="arc3,rad=0.06")
    add_arrow(ax, (0.745, 0.46), (0.815, 0.68), connectionstyle="arc3,rad=-0.10")
    add_arrow(ax, (0.745, 0.19), (0.815, 0.68), connectionstyle="arc3,rad=-0.20")
    add_arrow(ax, (0.225, 0.76), (0.815, 0.76))

    ax.text(
        0.5,
        0.025,
        (
            "Una imagen V1 por paciente y ausencia de solapamiento entre ejemplos "
            "y paciente reservado."
        ),
        ha="center",
        va="center",
        fontsize=8.0,
        color=MUTED,
    )

    for suffix in ("png", "pdf"):
        fig.savefig(
            args.output_dir / f"fig_k_examples_split.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
