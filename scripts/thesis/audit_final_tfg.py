#!/usr/bin/env python3
"""Audit final thesis claims and assets against the definitive LOOCV reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FINAL_ROOT = ROOT / "outputs" / "final_results_tfg"
DEFAULT_LOOCV_DIR = ROOT / "outputs" / "final_results_tfg" / "loocv"
DEFAULT_MIRROR_DIR = ROOT / "reports" / "loocv"
DEFAULT_THESIS_ROOT = ROOT / "thesis" / "thesis"
MODEL = "google/gemma-4-E4B-it"
ORIGINAL_SOURCE_SHA256 = "31F47D6BB91687F3B0F5B9B7532B6C58474CE34AB004485BB8BC05ABD9CF10DC"
CLINICAL_SOURCE_SHA256 = "73373A958C25A630425F5AE3B616CBC8D8AC1FE8829C9D44F3681C0CFF3B4561"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit final TFG results and assets.")
    parser.add_argument("--final-root", type=Path, default=DEFAULT_FINAL_ROOT)
    parser.add_argument("--loocv-dir", type=Path, default=DEFAULT_LOOCV_DIR)
    parser.add_argument("--mirror-dir", type=Path, default=DEFAULT_MIRROR_DIR)
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

    clinical_standard = read_csv(
        loocv_dir / "cnn_clinical" / "estandar" / "cnn_clinical_summary_by_k.csv"
    )
    clinical_balanced = read_csv(
        loocv_dir / "cnn_clinical" / "balanced" / "cnn_clinical_summary_by_k.csv"
    )
    expected_k_values = [2, 4, 8, 16, 32]
    for condition, rows in (
        ("estandar", clinical_standard),
        ("balanced", clinical_balanced),
    ):
        if sorted(int(row["k"]) for row in rows) != expected_k_values:
            raise AssertionError(f"Clinical CNN {condition}: unexpected K values")
        if any(int(row["n_patients"]) != 317 or int(row["n_runs"]) != 3 for row in rows):
            raise AssertionError(f"Clinical CNN {condition}: unexpected units")
    check_best(
        clinical_standard,
        condition="estandar",
        label="Clinical CNN standard",
        expected_k=32,
        expected_ba=0.5315234173957797,
    )
    check_best(
        clinical_balanced,
        condition="balanced",
        label="Clinical CNN balanced",
        expected_k=16,
        expected_ba=0.5456196031337565,
    )


def audit_canonical_package(final_root: Path, mirror_dir: Path) -> None:
    source_specs = (
        (
            final_root / "source" / "final_results_tfg.zip",
            final_root / "SOURCE.sha256",
            ORIGINAL_SOURCE_SHA256,
        ),
        (
            final_root / "source" / "cnn_clinical_brugada_loocv.zip",
            final_root / "SOURCE_CNN_CLINICAL.sha256",
            CLINICAL_SOURCE_SHA256,
        ),
    )
    for archive, checksum_file, expected in source_specs:
        actual = sha256(archive)
        recorded = checksum_file.read_text(encoding="utf-8").split()[0].upper()
        if actual != expected or recorded != expected:
            raise AssertionError(f"Unexpected source hash for {archive}")

    canonical_reports = final_root / "loocv" / "cnn_clinical"
    mirrored_reports = mirror_dir / "cnn_clinical"
    canonical_files = sorted(
        path.relative_to(canonical_reports)
        for path in canonical_reports.rglob("*")
        if path.is_file()
    )
    mirrored_files = sorted(
        path.relative_to(mirrored_reports) for path in mirrored_reports.rglob("*") if path.is_file()
    )
    if canonical_files != mirrored_files:
        raise AssertionError("Clinical CNN canonical reports and mirror contain different files")
    for relative_path in canonical_files:
        if sha256(canonical_reports / relative_path) != sha256(mirrored_reports / relative_path):
            raise AssertionError(f"Clinical CNN mirror differs: {relative_path}")

    prediction_files = sorted(
        (final_root / "runs" / "cnn_clinical_brugada_loocv").glob(
            "*/resnet18/k*_seed*/fold_predictions.jsonl"
        )
    )
    if len(prediction_files) != 30:
        raise AssertionError(f"Clinical CNN: expected 30 runs, found {len(prediction_files)}")
    for path in prediction_files:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != 317:
            raise AssertionError(f"Clinical CNN: expected 317 folds in {path}")
        k = int(re.search(r"k(\d+)_seed", path.parent.name).group(1))
        for record in records:
            test_patient = str(record["patient_id"])
            context_ids = split_ids(record.get("context_patient_ids", ""))
            validation_ids = split_ids(record.get("validation_patient_ids", ""))
            if len(context_ids) != k or len(validation_ids) != 8:
                raise AssertionError(f"Clinical CNN: unexpected split size in {path}")
            if test_patient in context_ids or test_patient in validation_ids:
                raise AssertionError(f"Clinical CNN: test-patient leakage in {path}")
            if set(context_ids).intersection(validation_ids):
                raise AssertionError(f"Clinical CNN: train-validation overlap in {path}")


def audit_thesis(thesis_root: Path) -> None:
    tex_files = sorted(thesis_root.rglob("*.tex"))
    joined = "\n".join(path.read_text(encoding="utf-8") for path in tex_files)
    if re.search(r"\bhallazgos?\b", joined, flags=re.IGNORECASE):
        raise AssertionError("The thesis still uses the prohibited word 'hallazgo'")
    forbidden_phrases = (
        r"\bcohort(?:e|es|s)?\b",
        r"\breferencia(?:s)?\s+(?:clínica(?:s)?\s+)?binaria(?:s)?\b",
        r"\bdominios?\s+sintéticos?\b",
        r"\bcontexto\s+sintético\b",
        r"\bCNN\s+(?:clínica|morfológica)\b",
        r"\btransductiv[oa]s?\b",
        r"\bparada\s+temprana\b",
        r"\bICL\s+(?:clínic[oa]|morfológic[oa])\b",
    )
    for pattern in forbidden_phrases:
        match = re.search(pattern, joined, flags=re.IGNORECASE)
        if match:
            raise AssertionError(f"Prohibited editorial phrase: {match.group()}")

    publication_files = [
        path
        for directory in ("chapters", "frontmatter", "appendices", "generated")
        for path in (thesis_root / directory).glob("*.tex")
    ]
    internal_report_terms = re.compile(
        r"\b(?:auditor[ií]as?|SHA(?:[- ]?256)?|hash(?:es)?|CSV)\b",
        flags=re.IGNORECASE,
    )
    for path in publication_files:
        match = internal_report_terms.search(path.read_text(encoding="utf-8"))
        if match:
            raise AssertionError(
                f"Internal reporting term in thesis prose: {path}: {match.group()}"
            )

    forbidden_dashes = re.compile(r"[\u2010-\u2015]")
    for path in tex_files:
        text = path.read_text(encoding="utf-8")
        match = forbidden_dashes.search(text)
        if match:
            raise AssertionError(f"Unicode dash in {path}: U+{ord(match.group()):04X}")

    drawio_assets = (
        "fig_pipeline_experimental.pdf",
        "gaussian_generator.drawio.pdf",
        "cnn_convolutional_pipeline.pdf",
        "vlm_icl_context_window.pdf",
        "fig_synthetic_real_comparison.pdf",
    )
    for asset in drawio_assets:
        count = joined.count(asset)
        if count != 1:
            raise AssertionError(f"{asset}: expected one TeX reference, found {count}")
    for removed_asset in ("fig_k_examples_split.pdf", "coral_alignment.drawio.pdf"):
        if removed_asset in joined:
            raise AssertionError(f"The removed diagram is still included: {removed_asset}")
    for asset in (
        "clinical_cnn_vs_gemma_performance.pdf",
        "clinical_cnn_vs_gemma_confusions.pdf",
    ):
        count = joined.count(asset)
        if count != 1:
            raise AssertionError(f"{asset}: expected one TeX reference, found {count}")

    graphics = re.findall(r"\\safeincludegraphics(?:\[[^]]*\])?\{([^}]+)\}", joined)
    non_pdf = [graphic for graphic in graphics if Path(graphic).suffix.lower() != ".pdf"]
    if non_pdf:
        raise AssertionError("Thesis figures must be referenced as PDF:\n" + "\n".join(non_pdf))
    preamble = (thesis_root / "config" / "preamble.tex").read_text(encoding="utf-8")
    direct_graphics = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", preamble)
    non_pdf_direct = [
        graphic
        for graphic in direct_graphics
        if Path(graphic).suffix and Path(graphic).suffix.lower() != ".pdf"
    ]
    if non_pdf_direct:
        raise AssertionError(
            "Thesis logos must be referenced as PDF:\n" + "\n".join(non_pdf_direct)
        )
    missing = []
    for graphic in graphics:
        path = thesis_root / "assets" / graphic
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise AssertionError("Missing thesis figures:\n" + "\n".join(missing))

    prose_dirs = ("chapters", "frontmatter", "appendices")
    semicolon_lines = [
        f"{path}:{number}"
        for directory in prose_dirs
        for path in (thesis_root / directory).glob("*.tex")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if ";" in line and not line.lstrip().startswith("%")
    ]
    if semicolon_lines:
        raise AssertionError("Semicolons in thesis prose:\n" + "\n".join(semicolon_lines))

    visual_sources = (thesis_root / "assets" / "results" / "fig_synthetic_real_comparison.drawio",)
    forbidden_visual_markers = ("Ventanas\\nV1/V2/V3", "Dataset real HUCA")
    for path in visual_sources:
        text = path.read_text(encoding="utf-8")
        for marker in forbidden_visual_markers:
            if marker in text:
                raise AssertionError(f"Obsolete visual marker in {path}: {marker}")

    gradcam_path = thesis_root / "assets" / "results" / "cnn_simulator_qrs_gradcam_example.pdf"
    width, height = pdf_dimensions(gradcam_path)
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
        "0,522",
        "300 imágenes únicas",
        "951 imágenes únicas",
        "Exactitud equilibrada",
        "La CNN no se reentrena con pacientes del HUCA",
        "The CNN is not retrained on HUCA patients",
        "outputs/work/cnn_clinical_brugada_loocv",
        "cnn_clinical_ba_by_k.pdf",
        "cnn_clinical_f1_by_k.pdf",
        "cnn_clinical_sens_spec_by_k.pdf",
        "comparison_huca_clinical_cnn_vs_gemma_ba_by_k.pdf",
        "comparison_huca_clinical_cnn_vs_gemma_f1_by_k.pdf",
        "cnn_clinical_sens_spec_best.pdf",
        "cnn_clinical_vs_gemma_confusion_matrices.pdf",
    )
    for marker in obsolete:
        if marker in joined:
            raise AssertionError(f"Obsolete final-report marker remains: {marker}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def audit_huca_provenance_report() -> None:
    report = json.loads((ROOT / "thesis/AUDITORIA_PROCEDENCIA_HUCA.json").read_text("utf-8"))
    public = report["public_metadata"]
    if public["label_counts"] != {"0": 287, "1": 69, "2": 7}:
        raise AssertionError("The public HUCA label counts require a new editorial review")
    if not public["publisher_checksum_verified"]:
        raise AssertionError("The public HUCA metadata checksum was not verified")
    if public["public_vs_local"]["different_labels"] != 93:
        raise AssertionError("Unexpected public/local HUCA label discrepancy")
    if public["public_vs_canonical"]["different_labels"] != 66:
        raise AssertionError("Unexpected public/canonical HUCA label discrepancy")
    if public["public_to_canonical_label_counts"] != {
        "0_to_0": 201,
        "0_to_1": 63,
        "1_to_1": 50,
        "2_to_1": 3,
    }:
        raise AssertionError("The HUCA provenance table does not match the audit report")
    expected_match = {
        "shared_patients": 317,
        "different_labels": 0,
        "only_left": 0,
        "only_right": 0,
    }
    for key in ("builder_vs_canonical", "processed_vs_canonical"):
        if report[key] != expected_match:
            raise AssertionError(f"Local HUCA selection does not match canonical runs: {key}")
    raw_metadata = ROOT / "data/raw/brugada-huca/1.0.0/metadata.csv"
    if raw_metadata.exists() and sha256(raw_metadata) != report["local_metadata_sha256"].upper():
        raise AssertionError("Local HUCA metadata changed after the provenance audit")


def split_ids(value: object) -> list[str]:
    return [item for item in str(value).split("|") if item]


def pdf_dimensions(path: Path) -> tuple[float, float]:
    match = re.search(
        rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
        path.read_bytes(),
    )
    if not match:
        raise AssertionError(f"Missing PDF MediaBox: {path}")
    return float(match.group(1)), float(match.group(2))


def main() -> None:
    args = parse_args()
    if not args.skip_results:
        audit_results(args.loocv_dir)
        audit_canonical_package(args.final_root, args.mirror_dir)
        audit_huca_provenance_report()
    audit_thesis(args.thesis_root)
    if args.skip_results:
        print("[OK] Final TFG text, Draw.io usage and figure references are consistent.")
    else:
        print("[OK] Final TFG results, units, Draw.io usage and figure references are consistent.")
        print("[WARNING] 66 HUCA labels differ from public CSV. Clinical provenance is unresolved.")


if __name__ == "__main__":
    main()
