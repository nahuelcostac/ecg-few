#!/usr/bin/env python3
"""Keep the V1 row of the synthetic Grad-CAM example for the thesis."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "thesis" / "thesis" / "assets" / "results"
SOURCE = ASSETS / "cnn_simulator_qrs_gradcam_example.png"
TARGET = ASSETS / "cnn_simulator_qrs_gradcam_example.pdf"


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    if image.size != (1680, 1536):
        raise ValueError(f"Unexpected Grad-CAM source dimensions: {image.size}")
    panels = (
        ("Entrada V1", (100, 110, 510, 525)),
        ("Mapa Grad-CAM", (615, 110, 1055, 525)),
        ("Superposición", (1140, 110, 1590, 525)),
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.25))
    for ax, (title, bounds) in zip(axes, panels, strict=False):
        ax.imshow(image.crop(bounds))
        ax.set_title(title, fontsize=11, pad=7)
        ax.axis("off")
    fig.subplots_adjust(left=0.012, right=0.988, top=0.88, bottom=0.02, wspace=0.045)
    fig.savefig(TARGET, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    main()
