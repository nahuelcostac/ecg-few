#!/usr/bin/env python3
"""Render an editable, vector diagram of the Deep CORAL training objective."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from drawio import Drawio
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[2]
WIDTH = 1400
HEIGHT = 560

INK = "#202124"
MUTED = "#5F6368"
LINE = "#7A7A7A"
SOURCE = "#D9912B"
SOURCE_FILL = "#FFF2DD"
TARGET = "#3F82C4"
TARGET_FILL = "#E8F2FC"
CORAL = "#735DA5"
CORAL_FILL = "#F0EBF8"
WHITE = "#FFFFFF"


def box_style(fill: str, stroke: str, *, size: int = 13, bold: bool = False) -> str:
    return (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=10;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=1.4;"
        f"fontFamily=Arial;fontSize={size};fontColor={INK};"
        f"fontStyle={1 if bold else 0};align=center;verticalAlign=middle;spacing=5;"
    )


def text_style(*, size: int = 13, bold: bool = False, color: str = INK) -> str:
    return (
        "text;html=1;strokeColor=none;fillColor=none;whiteSpace=wrap;"
        f"fontFamily=Arial;fontSize={size};fontColor={color};"
        f"fontStyle={1 if bold else 0};align=center;verticalAlign=middle;"
    )


def ellipse_style(fill: str, stroke: str, *, dashed: bool = False) -> str:
    return (
        "ellipse;whiteSpace=wrap;html=1;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=2;"
        f"dashed={1 if dashed else 0};"
    )


def add_drawio_edge(
    diagram: Drawio,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
    color: str = INK,
    width: float = 2.0,
) -> None:
    cell = ET.SubElement(
        diagram.root,
        "mxCell",
        {
            "id": diagram._id(),
            "value": "",
            "style": (
                "endArrow=block;html=1;rounded=0;"
                f"strokeWidth={width};strokeColor={color};dashed={1 if dashed else 0};"
            ),
            "edge": "1",
            "parent": "1",
        },
    )
    geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
    ET.SubElement(
        geometry,
        "mxPoint",
        {"x": str(start[0]), "y": str(start[1]), "as": "sourcePoint"},
    )
    ET.SubElement(
        geometry,
        "mxPoint",
        {"x": str(end[0]), "y": str(end[1]), "as": "targetPoint"},
    )


def rounded(
    axes,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    stroke: str,
    *,
    line_width: float = 1.4,
    radius: float = 12,
) -> None:
    axes.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius}",
            facecolor=fill,
            edgecolor=stroke,
            linewidth=line_width,
        )
    )


def label(axes, x: float, y: float, value: str, **kwargs) -> None:
    options = {
        "ha": "center",
        "va": "center",
        "fontfamily": "Arial",
        "fontsize": 12,
        "color": INK,
    }
    options.update(kwargs)
    axes.text(x, y, value, **options)


def arrow(
    axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
    color: str = INK,
    curved: float = 0.0,
) -> None:
    axes.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.8,
            linestyle="--" if dashed else "-",
            color=color,
            connectionstyle=f"arc3,rad={curved}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def point_grid(axes, x: float, y: float, color: str) -> None:
    for row in range(2):
        for column in range(3):
            axes.add_patch(
                Circle(
                    (x + column * 48, y + row * 48),
                    15,
                    facecolor=WHITE,
                    edgecolor=color,
                    linewidth=3.0,
                )
            )


def add_drawio_point_grid(diagram: Drawio, x: float, y: float, color: str) -> None:
    for row in range(2):
        for column in range(3):
            diagram.vertex(
                "",
                ellipse_style(WHITE, color),
                x + column * 48 - 15,
                y + row * 48 - 15,
                30,
                30,
            )


def draw_preview(path: Path) -> None:
    figure, axes = plt.subplots(figsize=(14, 5.6))
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    axes.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    axes.axis("off")

    rounded(axes, 35, 55, 245, 185, SOURCE_FILL, SOURCE)
    label(axes, 157.5, 84, "Origen: simulador", fontsize=14, fontweight="bold")
    label(axes, 157.5, 113, "Con etiquetas", fontsize=11, color=MUTED)
    point_grid(axes, 105, 155, SOURCE)

    rounded(axes, 35, 320, 245, 185, TARGET_FILL, TARGET)
    label(axes, 157.5, 349, "Objetivo: HUCA", fontsize=14, fontweight="bold")
    label(axes, 157.5, 378, "Sin etiquetas", fontsize=11, color=MUTED)
    point_grid(axes, 105, 420, TARGET)

    arrow(axes, (280, 148), (365, 220), color=SOURCE)
    arrow(axes, (280, 412), (365, 340), color=TARGET)

    rounded(axes, 365, 145, 230, 270, WHITE, LINE)
    label(axes, 480, 180, "Extractor compartido", fontsize=14, fontweight="bold")
    label(axes, 480, 207, "ResNet18", fontsize=11, color=MUTED)
    layer_colors = ("#F1F3F4", "#E3E7EA", "#D5DADF")
    for index, fill in enumerate(layer_colors):
        x = 407 + index * 26
        y = 250 + index * 20
        axes.add_patch(
            Polygon(
                [(x, y), (x + 98, y - 24), (x + 98, y + 73), (x, y + 97)],
                closed=True,
                facecolor=fill,
                edgecolor=INK,
                linewidth=1.2,
            )
        )
    label(axes, 480, 385, "mismos pesos", fontsize=10.5, color=MUTED)

    arrow(axes, (595, 280), (685, 280))
    rounded(axes, 685, 65, 345, 430, WHITE, LINE)
    label(axes, 857.5, 95, "Espacio de características", fontsize=14, fontweight="bold")
    label(
        axes, 857.5, 122, "representaciones del origen y del objetivo", fontsize=10.5, color=MUTED
    )

    axes.add_patch(
        Ellipse(
            (850, 280),
            250,
            155,
            angle=18,
            facecolor=SOURCE_FILL,
            edgecolor=SOURCE,
            linewidth=2.0,
            linestyle="--",
            alpha=0.72,
        )
    )
    axes.add_patch(
        Ellipse(
            (850, 280),
            185,
            250,
            angle=-28,
            facecolor=TARGET_FILL,
            edgecolor=TARGET,
            linewidth=2.0,
            linestyle="--",
            alpha=0.68,
        )
    )
    source_points = ((775, 230), (845, 210), (920, 245), (790, 315), (875, 300), (930, 350))
    target_points = ((820, 180), (810, 270), (900, 275), (825, 370), (900, 390), (950, 300))
    for x, y in source_points:
        axes.add_patch(Circle((x, y), 13, facecolor=WHITE, edgecolor=SOURCE, linewidth=3.0))
    for x, y in target_points:
        axes.add_patch(Circle((x, y), 13, facecolor=WHITE, edgecolor=TARGET, linewidth=3.0))
    label(axes, 850, 452, r"$C_s \approx C_t$", fontsize=17, fontweight="bold", color=CORAL)

    arrow(axes, (1030, 225), (1080, 225))
    rounded(axes, 1080, 110, 285, 145, WHITE, INK)
    label(axes, 1222.5, 143, "Objetivo conjunto", fontsize=13.5, fontweight="bold")
    label(
        axes,
        1222.5,
        190,
        r"$\mathcal{L}=\mathcal{L}_{sup}+\lambda\mathcal{L}_{CORAL}$",
        fontsize=16,
    )
    label(axes, 1222.5, 229, "clasificación + alineamiento", fontsize=10.5, color=MUTED)

    rounded(axes, 1080, 325, 285, 125, CORAL_FILL, CORAL)
    label(axes, 1222.5, 355, "Pérdida CORAL", fontsize=13.5, fontweight="bold", color=CORAL)
    label(axes, 1222.5, 395, r"$\Vert C_s-C_t\Vert_F^2$", fontsize=16, color=CORAL)
    label(axes, 1222.5, 427, "sin etiquetas del HUCA", fontsize=10.5, color=MUTED)

    arrow(axes, (1195, 450), (515, 415), dashed=True, color=CORAL, curved=0.24)
    figure.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(figure)


def write_drawio(path: Path) -> None:
    diagram = Drawio(width=WIDTH, height=HEIGHT, diagram_id="deep-coral", name="Deep CORAL")

    add_drawio_edge(diagram, (280, 148), (365, 220), color=SOURCE)
    add_drawio_edge(diagram, (280, 412), (365, 340), color=TARGET)
    add_drawio_edge(diagram, (595, 280), (685, 280))
    add_drawio_edge(diagram, (1030, 225), (1080, 225))
    add_drawio_edge(diagram, (1195, 450), (515, 415), dashed=True, color=CORAL)

    diagram.vertex("", box_style(SOURCE_FILL, SOURCE), 35, 55, 245, 185)
    diagram.vertex("Origen: simulador", text_style(size=14, bold=True), 55, 64, 205, 30)
    diagram.vertex("Con etiquetas", text_style(size=11, color=MUTED), 55, 94, 205, 25)
    add_drawio_point_grid(diagram, 105, 155, SOURCE)

    diagram.vertex("", box_style(TARGET_FILL, TARGET), 35, 320, 245, 185)
    diagram.vertex("Objetivo: HUCA", text_style(size=14, bold=True), 55, 329, 205, 30)
    diagram.vertex("Sin etiquetas", text_style(size=11, color=MUTED), 55, 359, 205, 25)
    add_drawio_point_grid(diagram, 105, 420, TARGET)

    diagram.vertex(
        "Extractor compartido<br><b>ResNet18</b><br><br>mismos pesos",
        box_style(WHITE, LINE, size=14, bold=False),
        365,
        145,
        230,
        270,
    )

    diagram.vertex("", box_style(WHITE, LINE), 685, 65, 345, 430)
    diagram.vertex("Espacio de características", text_style(size=14, bold=True), 705, 78, 305, 30)
    diagram.vertex(
        "representaciones del origen y del objetivo",
        text_style(size=11, color=MUTED),
        705,
        108,
        305,
        25,
    )
    diagram.vertex("", ellipse_style(SOURCE_FILL, SOURCE, dashed=True), 725, 205, 250, 155)
    diagram.vertex("", ellipse_style(TARGET_FILL, TARGET, dashed=True), 757, 155, 185, 250)
    for x, y in ((775, 230), (845, 210), (920, 245), (790, 315), (875, 300), (930, 350)):
        diagram.vertex("", ellipse_style(WHITE, SOURCE), x - 13, y - 13, 26, 26)
    for x, y in ((820, 180), (810, 270), (900, 275), (825, 370), (900, 390), (950, 300)):
        diagram.vertex("", ellipse_style(WHITE, TARGET), x - 13, y - 13, 26, 26)
    diagram.vertex("Cₛ ≈ Cₜ", text_style(size=17, bold=True, color=CORAL), 770, 430, 160, 35)

    diagram.vertex(
        "<b>Objetivo conjunto</b><br><br>L = Lsup + λ LCORAL<br><br>clasificación + alineamiento",
        box_style(WHITE, INK, size=13),
        1080,
        110,
        285,
        145,
    )
    diagram.vertex(
        "<b>Pérdida CORAL</b><br><br>||Cₛ - Cₜ||²F<br><br>sin etiquetas del HUCA",
        box_style(CORAL_FILL, CORAL, size=13),
        1080,
        325,
        285,
        125,
    )
    diagram.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "thesis/thesis/assets/results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    drawio_path = args.output_dir / "coral_alignment.drawio"
    pdf_path = args.output_dir / "coral_alignment.drawio.pdf"
    write_drawio(drawio_path)
    draw_preview(pdf_path)
    print(f"[OK] Rendered {drawio_path.name} and {pdf_path.name}.")


if __name__ == "__main__":
    main()
