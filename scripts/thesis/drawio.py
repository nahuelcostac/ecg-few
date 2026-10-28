"""Minimal helpers for creating editable Draw.io documents."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


class Drawio:
    def __init__(self, *, width: int, height: int, diagram_id: str, name: str) -> None:
        self.index = 2
        self.mxfile = ET.Element("mxfile", {"host": "Electron", "version": "29.6.1"})
        diagram = ET.SubElement(self.mxfile, "diagram", {"id": diagram_id, "name": name})
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
                "pageWidth": str(width),
                "pageHeight": str(height),
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
                "style": ("endArrow=block;html=1;rounded=0;strokeWidth=2;strokeColor=#5F6368;"),
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
