#!/usr/bin/env python3
"""Render the HUCA preprocessing diagram with patient-level counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "thesis" / "thesis" / "assets" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render HUCA preprocessing figure.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PNG and PDF figures are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(10.6, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.04, 0.91, "Preparación de la muestra Brugada-HUCA", fontsize=14, weight="bold")
    ax.plot([0.04, 0.96], [0.875, 0.875], color="#C7D1DE", linewidth=1.2)

    boxes = (
        (0.035, "WFDB +\nmetadatos", "363 registros\n12 derivaciones", "#E7F0FA", "#4E7BAE"),
        (
            0.225,
            "Criterios\nde entrada",
            "46 patrones basales\nlimítrofes excluidos",
            "#FFF3E8",
            "#F28C28",
        ),
        (
            0.415,
            "Detección\nde R",
            "V2 → V1 → V3 → II\napoyo para localizar R",
            "#E5F4F2",
            "#2E9991",
        ),
        (0.605, "Ventana\nfinal V1", "300 ms antes\n600 ms después", "#EAF5E9", "#5B9852"),
        (
            0.795,
            "Imágenes\ny pliegues",
            "317 imágenes V1\n317 pliegues LOOCV",
            "#F1EAF8",
            "#8A62B8",
        ),
    )
    width = 0.17
    height = 0.235
    y = 0.475
    for index, (x, title, body, fill, edge) in enumerate(boxes):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.35,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + 0.166,
            title,
            ha="center",
            va="center",
            fontsize=8.4,
            weight="bold",
            linespacing=0.9,
        )
        ax.text(
            x + width / 2,
            y + 0.066,
            body,
            ha="center",
            va="center",
            fontsize=7.5,
            color="#333333",
        )
        if index < len(boxes) - 1:
            next_x = boxes[index + 1][0]
            arrow = FancyArrowPatch(
                (x + width, y + height / 2),
                (next_x, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.6,
                color="#727A84",
            )
            ax.add_patch(arrow)

    ax.text(
        0.05,
        0.30,
        "Solo la ventana V1 se renderiza como entrada del modelo.",
        fontsize=9.4,
        weight="bold",
    )
    ax.text(
        0.05,
        0.21,
        (
            "La referencia clínica clinical_brugada se conserva por paciente; "
            "tres semillas producen 951 predicciones."
        ),
        fontsize=8.6,
        color="#333333",
    )

    for suffix in ("png", "pdf"):
        fig.savefig(
            args.output_dir / f"fig_huca_preprocesado.{suffix}",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
