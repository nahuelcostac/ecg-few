#!/usr/bin/env python3
"""Render an editable V1-only ICL diagram and its vector PDF for the thesis."""

from __future__ import annotations

import base64
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "thesis" / "thesis" / "assets" / "results"
INK = "#202124"
BLUE = "#DBE8FB"
GREEN = "#D9EAD3"
PURPLE = "#EADCF4"


def wave(x: np.ndarray, variant: float = 0.0) -> np.ndarray:
    return (
        0.10 * np.exp(-(((x - 0.19) / 0.045) ** 2))
        - 0.13 * np.exp(-(((x - 0.40) / 0.012) ** 2))
        + (0.70 + variant) * np.exp(-(((x - 0.43) / 0.014) ** 2))
        - 0.25 * np.exp(-(((x - 0.47) / 0.025) ** 2))
        + 0.16 * np.exp(-(((x - 0.72) / 0.09) ** 2))
    )


def ecg_svg(variant: float) -> str:
    x = np.linspace(0, 1, 200)
    points = " ".join(
        f"{4 + 168 * t:.1f},{34 - 28 * v:.1f}" for t, v in zip(x, wave(x, variant), strict=False)
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="180" height="55" viewBox="0 0 180 55">'
        '<rect width="180" height="55" fill="#fffafa"/>'
        '<path d="M0 15H180M0 35H180M20 0V55M60 0V55M100 0V55M140 0V55" '
        'stroke="#f0caca" stroke-width="0.7"/>'
        f'<polyline points="{points}" fill="none" stroke="#202124" stroke-width="2"/>'
        "</svg>"
    )
    return base64.b64encode(svg.encode("utf-8")).decode("ascii")


def write_drawio() -> None:
    mxfile = ET.Element("mxfile", {"host": "Electron", "version": "29.6.1"})
    diagram = ET.SubElement(mxfile, "diagram", {"id": "icl-v1", "name": "Contexto ICL V1"})
    model = ET.SubElement(
        diagram,
        "mxGraphModel",
        {"page": "1", "pageWidth": "1150", "pageHeight": "355", "grid": "1", "guides": "1"},
    )
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", {"id": "0"})
    ET.SubElement(root, "mxCell", {"id": "1", "parent": "0"})
    index = 2

    def vertex(value: str, style: str, x: int, y: int, w: int, h: int) -> None:
        nonlocal index
        cell = ET.SubElement(
            root,
            "mxCell",
            {"id": str(index), "value": value, "style": style, "vertex": "1", "parent": "1"},
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {"x": str(x), "y": str(y), "width": str(w), "height": str(h), "as": "geometry"},
        )
        index += 1

    def edge(x1: int, y1: int, x2: int, y2: int) -> None:
        nonlocal index
        cell = ET.SubElement(
            root,
            "mxCell",
            {
                "id": str(index),
                "style": "endArrow=block;html=1;strokeWidth=2;strokeColor=#5F6368;",
                "edge": "1",
                "parent": "1",
            },
        )
        geometry = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        ET.SubElement(geometry, "mxPoint", {"x": str(x1), "y": str(y1), "as": "sourcePoint"})
        ET.SubElement(geometry, "mxPoint", {"x": str(x2), "y": str(y2), "as": "targetPoint"})
        index += 1

    box = (
        "rounded=1;whiteSpace=wrap;html=1;arcSize=8;strokeWidth=1.5;"
        "fontFamily=Arial;fontSize=16;fontColor=#202124;align=center;verticalAlign=top;spacingTop=12;"
    )
    vertex(
        "Instrucción común sobre la imagen V1",
        box + "fillColor=#F4F4F4;strokeColor=#B8B8B8;",
        60,
        15,
        720,
        48,
    )
    for title, x, fill, variant in [
        ("Demostración 1", 60, BLUE, 0.0),
        ("Demostración K", 335, BLUE, -0.10),
        ("Consulta V1", 610, GREEN, 0.07),
    ]:
        vertex(title, box + f"fillColor={fill};strokeColor=#8C9EAD;", x, 96, 220, 150)
        vertex(
            "",
            "shape=image;aspect=fixed;html=1;image=data:image/svg+xml;base64,"
            + ecg_svg(variant)
            + ";",
            x + 20,
            143,
            180,
            55,
        )
        vertex(
            "Etiqueta conocida" if x < 610 else "Etiqueta pendiente",
            "text;html=1;align=center;verticalAlign=middle;fontFamily=Arial;fontSize=13;",
            x + 14,
            205,
            192,
            28,
        )
    vertex("…", "text;html=1;align=center;fontSize=28;fontColor=#5F6368;", 285, 152, 46, 45)
    vertex(
        "Gemma 4<br><font color='#5F6368'>pesos fijos</font>",
        box + f"fillColor={PURPLE};strokeColor=#9673A6;",
        890,
        108,
        200,
        92,
    )
    vertex("Predicción", box + "fillColor=#F4F4F4;strokeColor=#B8B8B8;", 890, 228, 200, 58)
    edge(830, 172, 890, 154)
    edge(990, 200, 990, 228)
    ET.indent(mxfile, space="  ")
    ET.ElementTree(mxfile).write(OUT / "vlm_icl_context_window.drawio", encoding="utf-8")


def pdf_box(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str, fill: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=fill,
            edgecolor="#8C9EAD",
            lw=1.1,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.08,
        title,
        ha="center",
        va="top",
        fontsize=10.5,
        weight="bold",
        color=INK,
    )


def write_pdf() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 3.3))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 3.5)
    ax.axis("off")
    pdf_box(ax, 0.4, 2.82, 7.7, 0.42, "Instrucción común sobre la imagen V1", "#F4F4F4")
    for title, x0, fill, variant, label in [
        ("Demostración 1", 0.4, BLUE, 0.0, "Etiqueta conocida"),
        ("Demostración K", 3.4, BLUE, -0.10, "Etiqueta conocida"),
        ("Consulta V1", 6.4, GREEN, 0.07, "Etiqueta pendiente"),
    ]:
        pdf_box(ax, x0, 0.58, 2.55, 1.96, title, fill)
        for gx in np.linspace(x0 + 0.22, x0 + 2.32, 11):
            ax.plot([gx, gx], [1.12, 2.01], color="#EECACA", lw=0.35, zorder=1)
        for gy in np.linspace(1.12, 2.01, 6):
            ax.plot([x0 + 0.22, x0 + 2.32], [gy, gy], color="#EECACA", lw=0.35, zorder=1)
        xx = np.linspace(0, 1, 350)
        ax.plot(
            x0 + 0.22 + 2.10 * xx,
            1.52 + 0.58 * wave(xx, variant),
            color=INK,
            lw=1.65,
            zorder=2,
        )
        ax.text(x0 + 1.275, 0.83, label, ha="center", va="center", fontsize=9, color=INK)
    ax.text(3.18, 1.52, "…", ha="center", va="center", fontsize=20, color="#5F6368")
    pdf_box(ax, 9.43, 1.62, 1.67, 0.91, "Gemma 4", PURPLE)
    ax.text(10.27, 1.85, "pesos fijos", ha="center", va="center", fontsize=8.5, color="#5F6368")
    pdf_box(ax, 9.43, 0.52, 1.67, 0.68, "Predicción", "#F4F4F4")
    ax.add_patch(
        FancyArrowPatch(
            (8.98, 1.58),
            (9.40, 2.07),
            arrowstyle="-|>",
            mutation_scale=13,
            color="#5F6368",
            lw=1.2,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (10.27, 1.59),
            (10.27, 1.23),
            arrowstyle="-|>",
            mutation_scale=13,
            color="#5F6368",
            lw=1.2,
        )
    )
    fig.savefig(OUT / "vlm_icl_context_window.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_drawio()
    write_pdf()


if __name__ == "__main__":
    main()
