#!/usr/bin/env python3
"""Render a black-and-white flowchart for one direct clinical comparison fold."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from drawio import Drawio
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[2]
WIDTH = 1400
HEIGHT = 720
INK = "#111111"
WHITE = "#FFFFFF"
MUTED = "#444444"


def flow_style(shape: str = "rectangle", *, bold: bool = False, size: int = 14) -> str:
    font_style = 1 if bold else 0
    common = (
        "whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#111111;"
        f"strokeWidth=1.5;fontFamily=Arial;fontSize={size};fontColor=#111111;"
        f"fontStyle={font_style};spacing=6;"
    )
    if shape == "parallelogram":
        return "shape=parallelogram;perimeter=parallelogramPerimeter;" + common
    if shape == "rhombus":
        return "rhombus;" + common
    return "rounded=0;" + common


def add_line(
    diagram: Drawio, start: tuple[float, float], end: tuple[float, float], *, arrow: bool = False
) -> None:
    if arrow:
        diagram.edge(start, end)
        return
    cell = ET.SubElement(
        diagram.root,
        "mxCell",
        {
            "id": diagram._id(),
            "value": "",
            "style": "endArrow=none;html=1;rounded=0;strokeWidth=1.8;strokeColor=#111111;",
            "edge": "1",
            "parent": "1",
        },
    )
    geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    ET.SubElement(
        geometry, "mxPoint", {"x": str(start[0]), "y": str(start[1]), "as": "sourcePoint"}
    )
    ET.SubElement(geometry, "mxPoint", {"x": str(end[0]), "y": str(end[1]), "as": "targetPoint"})


def add_shape(
    diagram: Drawio,
    axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    shape: str = "rectangle",
    bold: bool = False,
    size: float = 13,
    label_size: int = 14,
) -> None:
    diagram.vertex(
        text.replace("\n", "<br>"), flow_style(shape, bold=bold, size=label_size), x, y, w, h
    )
    if shape == "parallelogram":
        skew = min(34, w * 0.12)
        points = [(x + skew, y), (x + w, y), (x + w - skew, y + h), (x, y + h)]
        axes.add_patch(
            Polygon(points, closed=True, facecolor=WHITE, edgecolor=INK, linewidth=1.4, zorder=3)
        )
    elif shape == "rhombus":
        points = [(x + w / 2, y), (x + w, y + h / 2), (x + w / 2, y + h), (x, y + h / 2)]
        axes.add_patch(
            Polygon(points, closed=True, facecolor=WHITE, edgecolor=INK, linewidth=1.4, zorder=3)
        )
    else:
        axes.add_patch(
            Rectangle((x, y), w, h, facecolor=WHITE, edgecolor=INK, linewidth=1.4, zorder=3)
        )
    axes.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontfamily="Arial",
        weight="bold" if bold else "normal",
        color=INK,
        linespacing=1.2,
        zorder=4,
    )


def add_arrow(diagram: Drawio, axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    add_line(diagram, start, end, arrow=True)
    axes.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13, color=INK, linewidth=1.25, zorder=2
        )
    )


def add_plain_line(
    diagram: Drawio, axes, start: tuple[float, float], end: tuple[float, float]
) -> None:
    add_line(diagram, start, end)
    axes.plot([start[0], end[0]], [start[1], end[1]], color=INK, linewidth=1.25, zorder=2)


def add_polyline_arrow(diagram: Drawio, axes, points: list[tuple[float, float]]) -> None:
    """Draw one continuous connector and place its only arrowhead at the end."""
    for start, end in zip(points[:-2], points[1:-1], strict=False):
        add_plain_line(diagram, axes, start, end)
    add_arrow(diagram, axes, points[-2], points[-1])


def add_labeled_arrow(
    diagram: Drawio,
    axes,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    label_pos: tuple[float, float],
) -> None:
    add_arrow(diagram, axes, start, end)
    axes.text(
        label_pos[0],
        label_pos[1],
        label,
        ha="center",
        va="center",
        fontsize=10.5,
        fontfamily="Arial",
        color=MUTED,
        zorder=4,
    )
    diagram.vertex(
        label,
        "text;html=1;align=center;fontFamily=Arial;fontSize=11;fontColor=#444444;",
        label_pos[0] - 45,
        label_pos[1] - 13,
        90,
        24,
    )


def add_labeled_polyline_arrow(
    diagram: Drawio,
    axes,
    points: list[tuple[float, float]],
    label: str,
    label_pos: tuple[float, float],
) -> None:
    add_polyline_arrow(diagram, axes, points)
    axes.text(
        label_pos[0],
        label_pos[1],
        label,
        ha="center",
        va="center",
        fontsize=10.5,
        fontfamily="Arial",
        color=MUTED,
        zorder=4,
    )
    diagram.vertex(
        label,
        "text;html=1;align=center;fontFamily=Arial;fontSize=11;fontColor=#444444;",
        label_pos[0] - 45,
        label_pos[1] - 13,
        90,
        24,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "thesis/thesis/assets/results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    diagram = Drawio(
        width=WIDTH,
        height=HEIGHT,
        diagram_id="clinical-fold",
        name="Comparación clínica",
    )

    figure, axes = plt.subplots(figsize=(14, 7.2))
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    axes.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    axes.axis("off")

    add_shape(
        diagram,
        axes,
        500,
        70,
        400,
        75,
        "Conjunto de datos del HUCA\n317 pacientes",
        shape="parallelogram",
        bold=True,
        size=15,
    )
    add_plain_line(diagram, axes, (700, 145), (700, 220))
    add_polyline_arrow(diagram, axes, [(700, 220), (250, 220), (250, 350)])
    add_shape(diagram, axes, 100, 350, 300, 80, "316 pacientes\ndisponibles", bold=True, size=14)
    add_polyline_arrow(diagram, axes, [(700, 220), (1235, 220), (1235, 350)])

    add_plain_line(diagram, axes, (250, 430), (430, 430))
    add_polyline_arrow(diagram, axes, [(430, 430), (430, 306), (470, 306)])
    add_polyline_arrow(diagram, axes, [(430, 430), (430, 446), (470, 446)])

    add_shape(
        diagram,
        axes,
        470,
        270,
        260,
        72,
        "CNN clínica\nK entrenamiento + 8 validación",
        bold=False,
        size=12.8,
    )
    add_shape(
        diagram, axes, 790, 270, 260, 72, "Entrenar CNN\nseleccionar época y umbral", size=12.8
    )
    add_arrow(diagram, axes, (730, 306), (790, 306))

    add_shape(
        diagram,
        axes,
        470,
        410,
        260,
        72,
        "Gemma 4 / ICL\nK demostraciones etiquetadas",
        bold=False,
        size=12.8,
    )
    add_shape(
        diagram, axes, 790, 410, 260, 72, "Preparar consulta\ninstrucciones + ejemplos", size=12.8
    )
    add_arrow(diagram, axes, (730, 446), (790, 446))

    add_shape(diagram, axes, 1110, 350, 250, 80, "Paciente\nreservado", bold=True, size=14)
    add_labeled_polyline_arrow(
        diagram,
        axes,
        [(1050, 306), (1080, 306), (1080, 370), (1110, 370)],
        "Predicción",
        (1080, 255),
    )
    add_labeled_polyline_arrow(
        diagram,
        axes,
        [(1050, 446), (1080, 446), (1080, 410), (1110, 410)],
        "Predicción",
        (1080, 505),
    )

    add_plain_line(diagram, axes, (1235, 430), (1235, 564))
    add_shape(
        diagram,
        axes,
        520,
        530,
        500,
        68,
        "Evaluación\ncomparar con la etiqueta reservada",
        bold=True,
        size=14,
    )
    add_arrow(diagram, axes, (1235, 564), (1020, 564))

    add_arrow(diagram, axes, (770, 598), (770, 625))
    add_shape(
        diagram,
        axes,
        620,
        625,
        300,
        72,
        "¿Completados los\n317 pliegues?",
        shape="rhombus",
        bold=True,
        size=13,
    )
    add_shape(diagram, axes, 1040, 630, 290, 62, "Resultados", bold=True, size=13)
    add_arrow(diagram, axes, (920, 661), (1040, 661))
    axes.text(980, 648, "Sí", ha="center", va="center", fontsize=11, fontfamily="Arial", color=INK)
    diagram.vertex(
        "Sí",
        "text;html=1;align=center;fontFamily=Arial;fontSize=11;fontColor=#111111;",
        955,
        637,
        45,
        24,
    )

    add_polyline_arrow(diagram, axes, [(620, 661), (35, 661), (35, 390), (100, 390)])
    axes.text(585, 648, "No", ha="center", va="center", fontsize=11, fontfamily="Arial", color=INK)
    diagram.vertex(
        "No",
        "text;html=1;align=center;fontFamily=Arial;fontSize=11;fontColor=#111111;",
        565,
        637,
        40,
        24,
    )

    base = args.output_dir / "clinical_fold_protocol"
    diagram.save(base.with_suffix(".drawio"))
    figure.savefig(base.with_suffix(".pdf"))
    figure.savefig(base.with_suffix(".png"), dpi=240)
    plt.close(figure)
    print(f"[OK] Rendered {base.name}: editable Draw.io, PDF and PNG.")


if __name__ == "__main__":
    main()
