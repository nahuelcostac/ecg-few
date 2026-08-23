#!/usr/bin/env python3
"""Audit final thesis claims and assets against the definitive LOOCV reports."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOOCV_DIR = ROOT / "reports" / "loocv"
DEFAULT_THESIS_ROOT = ROOT / "thesis" / "thesis"
MODEL = "google/gemma-4-E4B-it"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit final TFG results and assets.")
    parser.add_argument("--loocv-dir", type=Path, default=DEFAULT_LOOCV_DIR)
    parser.add_argument("--thesis-root", type=Path, default=DEFAULT_THESIS_ROOT)
    parser.add_argument(
        "--skip-results",
        action="store_true",
        help="Audit the thesis text and versioned assets without local LOOCV reports.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=5e-7):
        raise AssertionError(f"{label}: expected {expected}, found {actual}")


def best(rows: list[dict[str, str]], *, condition: str | None = None) -> dict[str, str]:
    selected = [row for row in rows if condition is None or row.get("condition") == condition]
    return max(selected, key=lambda row: float(row["balanced_accuracy_mean"]))


def check_best(
    rows: list[dict[str, str]],
    *,
    label: str,
    expected_k: int,
    expected_ba: float,
    condition: str | None = None,
) -> None:
    row = best(rows, condition=condition)
    if int(row["k"]) != expected_k:
        raise AssertionError(f"{label}: expected K={expected_k}, found K={row['k']}")
    assert_close(float(row["balanced_accuracy_mean"]), expected_ba, label)


def audit_results(loocv_dir: Path) -> None:
    check_best(
        read_csv(loocv_dir / "cnn_simulator_qrs" / "cnn_summary_by_k.csv"),
        label="CNN synthetic",
        expected_k=32,
        expected_ba=0.8854166666666666,
    )
    cnn_real = read_csv(loocv_dir / "cnn" / "cnn_summary_by_k.csv")
    check_best(
        cnn_real,
        label="CNN HUCA",
        expected_k=4,
        expected_ba=0.55118802539029,
    )

    domain = read_csv(
        loocv_dir / "cnn_domain_adaptation" / "comparison" / "cnn_domain_adaptation_by_k.csv"
    )
    mmd_32 = next(row for row in domain if row["method"] == "mmd" and int(row["k"]) == 32)
    assert_close(
        float(mmd_32["balanced_accuracy_mean"]),
        0.5455481214616572,
        "MMD K=32",
    )
    if float(mmd_32["balanced_accuracy_mean"]) >= float(best(cnn_real)["balanced_accuracy_mean"]):
        raise AssertionError("MMD K=32 must not be reported above the best base CNN")

    vlm_specs = (
        ("vlm_simulator_qrs", "VLM synthetic", 32, 0.9041666666666667),
        ("vlm_real_context", "VLM clinical context", 32, 0.7612369188540059),
        ("vlm", "VLM synthetic context", 16, 0.5141962600789157),
    )
    for folder, label, expected_k, expected_ba in vlm_specs:
        rows = [
            row
            for row in read_csv(loocv_dir / folder / "vlm_summary_by_model_condition_k.csv")
            if row["model"] == MODEL
        ]
        check_best(
            rows,
            condition="balanced",
            label=label,
            expected_k=expected_k,
            expected_ba=expected_ba,
        )

    validation_specs = (
        ("vlm_simulator_qrs", 100, 100),
        ("vlm_real_context", 317, 317),
        ("vlm", 317, 317),
    )
    for folder, patients, images in validation_specs:
        path = loocv_dir / folder / "vlm_setup_validation.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["patients"] != patients or payload["images"] != images:
            raise AssertionError(f"Unexpected unit counts in {path}")


def audit_thesis(thesis_root: Path) -> None:
    tex_files = sorted(thesis_root.rglob("*.tex"))
    joined = "\n".join(path.read_text(encoding="utf-8") for path in tex_files)

    forbidden_dashes = re.compile(r"[\u2010-\u2015]")
    for path in tex_files:
        text = path.read_text(encoding="utf-8")
        match = forbidden_dashes.search(text)
        if match:
            raise AssertionError(f"Unicode dash in {path}: U+{ord(match.group()):04X}")

    drawio_assets = (
        "fig_pipeline_experimental.drawio.png",
        "gaussian_generator.drawio.png",
        "cnn_convolutional_pipeline.drawio.png",
        "vlm_icl_context_window.drawio.png",
        "fig_synthetic_real_comparison.drawio.png",
    )
    for asset in drawio_assets:
        count = joined.count(asset)
        if count != 1:
            raise AssertionError(f"{asset}: expected one TeX reference, found {count}")

    graphics = re.findall(r"\\safeincludegraphics(?:\[[^]]*\])?\{([^}]+)\}", joined)
    missing = []
    for graphic in graphics:
        path = thesis_root / "assets" / graphic
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise AssertionError("Missing thesis figures:\n" + "\n".join(missing))

    visual_sources = (
        ROOT / "scripts" / "thesis" / "render_huca_preprocessing_figure.py",
        ROOT / "scripts" / "thesis" / "render_k_examples_split_figure.py",
        thesis_root / "assets" / "results" / "fig_synthetic_real_comparison.drawio",
    )
    forbidden_visual_markers = ("Ventanas\\nV1/V2/V3", "Dataset real HUCA")
    for path in visual_sources:
        text = path.read_text(encoding="utf-8")
        for marker in forbidden_visual_markers:
            if marker in text:
                raise AssertionError(f"Obsolete visual marker in {path}: {marker}")

    gradcam_path = thesis_root / "assets" / "results" / "cnn_simulator_qrs_gradcam_example.png"
    width, height = png_dimensions(gradcam_path)
    if width / height < 2.5:
        raise AssertionError(
            "The thesis Grad-CAM example must contain one V1 row with three panels"
        )

    figure_blocks = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", joined, flags=re.DOTALL)
    figure_labels = []
    for block in figure_blocks:
        match = re.search(r"\\label\{([^}]+)\}", block)
        if match:
            figure_labels.append(match.group(1))
    uncited_figures = []
    for label in figure_labels:
        reference = rf"\\(?:ref|pageref|autoref|cref|Cref)\{{{re.escape(label)}\}}"
        if not re.search(reference, joined):
            uncited_figures.append(label)
    if uncited_figures:
        raise AssertionError(
            "Figures without an explicit text reference:\n" + "\n".join(uncited_figures)
        )

    obsolete = (
        "0,848",
        "0,754",
        "0,549",
        "0,522",
        "300 imágenes únicas",
        "951 imágenes únicas",
        "Exactitud equilibrada",
    )
    for marker in obsolete:
        if marker in joined:
            raise AssertionError(f"Obsolete final-report marker remains: {marker}")


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise AssertionError(f"Invalid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def main() -> None:
    args = parse_args()
    if not args.skip_results:
        audit_results(args.loocv_dir)
    audit_thesis(args.thesis_root)
    if args.skip_results:
        print("[OK] Final TFG text, Draw.io usage and figure references are consistent.")
    else:
        print("[OK] Final TFG results, units, Draw.io usage and figure references are consistent.")


if __name__ == "__main__":
    main()
