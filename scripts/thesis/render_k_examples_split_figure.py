#!/usr/bin/env python3
"""Create the editable draw.io patient-split diagram and its rendered assets."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "thesis" / "thesis" / "assets" / "results"

INK = "#202124"
MUTED = "#5F6368"
BLUE_FILL = "#DBE8FB"
BLUE_STROKE = "#7D94B5"
RED_FILL = "#F4CCCC"
RED_STROKE = "#C16A61"
ORANGE_FILL = "#FCE5CD"
ORANGE_STROKE = "#D6A44A"
GREEN_FILL = "#D9EAD3"
GREEN_STROKE = "#93C47D"
PURPLE_FILL = "#EADCF4"
PURPLE_STROKE = "#A07BB8"
GRAY_FILL = "#F3F3F3"
GRAY_STROKE = "#8A8A8A"

WIDTH = 1180
HEIGHT = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the K-example split diagram.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the draw.io, PNG and PDF files are written.",
    )
    return parser.parse_args()


class Drawio:
    def __init__(self) -> None:
        self.index = 2
        self.mxfile = ET.Element("mxfile", {"host": "Electron", "version": "29.6.1"})
        diagram = ET.SubElement(
            self.mxfile,
            "diagram",
            {"id": "patient-split", "name": "Separacion por paciente"},
        )
        model = ET.SubElement(
            diagram,
            "mxGraphModel",
            {
                "dx": "1260",
                "dy": "720",
                "grid": "1",
                "gridSize": "10",
                "guides": "1",
                "tooltips": "1",
                "connect": "1",
                "arrows": "1",
                "fold": "1",
                "page": "1",
                "pageScale": "1",
                "pageWidth": str(WIDTH),
                "pageHeight": str(HEIGHT),
                "math": "0",
                "shadow": "0",
            },
        )
        self.root = ET.SubElement(model, "root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})

    def _id(self) -> str:
        value = str(self.index)
        self.index += 1
        return value

    def vertex(self, value: str, style: str, x: float, y: float, w: float, h: float) -> None:
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": self._id(),
                "value": value,
                "style": style,
                "vertex": "1",
                "parent": "1",
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
        )

    def edge(self, start: tuple[float, float], end: tuple[float, float]) -> None:
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": self._id(),
                "value": "",
                "style": "endArrow=block;html=1;rounded=0;strokeWidth=2;strokeColor=#5F6368;",
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

    def save(self, path: Path) -> None:
        ET.indent(self.mxfile, space="  ")
        ET.ElementTree(self.mxfile).write(path, encoding="utf-8", xml_declaration=False)


def box_style(fill: str, stroke: str) -> str:
    return (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;"
        f"fillColor={fill};strokeColor={stroke};strokeWidth=1.2;"
        "fontFamily=Arial;fontSize=12;fontColor=#202124;"
        "align=center;verticalAlign=middle;spacing=8;"
    )


def write_drawio(output_dir: Path) -> None:
    diagram = Drawio()

    diagram.edge((260, 112), (940, 112))
    diagram.edge((260, 350), (340, 350))
    diagram.edge((560, 330), (650, 270))
    diagram.edge((560, 370), (650, 400))
    diagram.edge((870, 270), (940, 150))
    diagram.edge((870, 400), (940, 150))

    diagram.vertex(
        "<b>Paciente <i>i</i> reservado</b><br><font color='#5F6368'>consulta o prueba</font>",
        box_style(RED_FILL, RED_STROKE),
        40,
        57,
        220,
        110,
    )
    diagram.vertex(
        "<b>Banco de ejemplos</b><br><font color='#5F6368'>pacientes disponibles<br>sin el paciente <i>i</i></font>",
        box_style(BLUE_FILL, BLUE_STROKE),
        40,
        285,
        220,
        130,
    )
    diagram.vertex(
        "<b>Selección de <i>K</i> ejemplos</b><br><font color='#5F6368'><i>K</i> = 2, 4, 8, 16, 32<br>tres semillas</font>",
        box_style(ORANGE_FILL, ORANGE_STROKE),
        340,
        285,
        220,
        130,
    )
    diagram.vertex(
        "<b>CNN</b><br><font color='#5F6368'>los ejemplos actualizan<br>los pesos</font>",
        box_style(GREEN_FILL, GREEN_STROKE),
        650,
        220,
        220,
        100,
    )
    diagram.vertex(
        "<b>VLM/ICL</b><br><font color='#5F6368'>los ejemplos forman<br>el contexto</font>",
        box_style(PURPLE_FILL, PURPLE_STROKE),
        650,
        350,
        220,
        100,
    )
    diagram.vertex(
        "<b>Evaluación</b><br><font color='#5F6368'>predicción sobre<br>el paciente <i>i</i></font>",
        box_style(GRAY_FILL, GRAY_STROKE),
        940,
        57,
        200,
        110,
    )

    diagram.save(output_dir / "fig_k_examples_split.drawio")


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
        boxstyle="round,pad=0.012,rounding_size=12",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height * 0.36,
        title,
        ha="center",
        va="center",
        fontsize=10.2,
        weight="bold",
        color=INK,
    )
    ax.text(
        x + width / 2,
        y + height * 0.70,
        body,
        ha="center",
        va="center",
        fontsize=8.7,
        color=MUTED,
        linespacing=1.15,
    )


def add_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.3,
            color=MUTED,
            connectionstyle="arc3",
        )
    )


def render_assets(output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(11.8, 5.0))
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis("off")

    add_arrow(ax, (260, 112), (940, 112))
    add_arrow(ax, (260, 350), (340, 350))
    add_arrow(ax, (560, 330), (650, 270))
    add_arrow(ax, (560, 370), (650, 400))
    add_arrow(ax, (870, 270), (940, 150))
    add_arrow(ax, (870, 400), (940, 150))

    add_box(
        ax,
        x=40,
        y=57,
        width=220,
        height=110,
        title=r"Paciente $i$ reservado",
        body="consulta o prueba",
        fill=RED_FILL,
        edge=RED_STROKE,
    )
    add_box(
        ax,
        x=40,
        y=285,
        width=220,
        height=130,
        title="Banco de ejemplos",
        body="pacientes disponibles\nsin el paciente i",
        fill=BLUE_FILL,
        edge=BLUE_STROKE,
    )
    add_box(
        ax,
        x=340,
        y=285,
        width=220,
        height=130,
        title=r"Selección de $K$ ejemplos",
        body=r"$K = 2, 4, 8, 16, 32$" + "\ntres semillas",
        fill=ORANGE_FILL,
        edge=ORANGE_STROKE,
    )
    add_box(
        ax,
        x=650,
        y=220,
        width=220,
        height=100,
        title="CNN",
        body="los ejemplos actualizan\nlos pesos",
        fill=GREEN_FILL,
        edge=GREEN_STROKE,
    )
    add_box(
        ax,
        x=650,
        y=350,
        width=220,
        height=100,
        title="VLM/ICL",
        body="los ejemplos forman\nel contexto",
        fill=PURPLE_FILL,
        edge=PURPLE_STROKE,
    )
    add_box(
        ax,
        x=940,
        y=57,
        width=200,
        height=110,
        title="Evaluación",
        body="predicción sobre\nel paciente i",
        fill=GRAY_FILL,
        edge=GRAY_STROKE,
    )

    for filename in ("fig_k_examples_split.drawio.png", "fig_k_examples_split.pdf"):
        fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_drawio(args.output_dir)
    render_assets(args.output_dir)
    print(args.output_dir / "fig_k_examples_split.drawio")
    print(args.output_dir / "fig_k_examples_split.drawio.png")
    print(args.output_dir / "fig_k_examples_split.pdf")


if __name__ == "__main__":
    main()
