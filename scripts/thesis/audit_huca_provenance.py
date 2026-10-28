#!/usr/bin/env python3
"""Compare HUCA metadata with canonical labels without publishing patient identifiers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_ROOT = "https://physionet.org/files/brugada-huca/1.0.0/"


def read_rows(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    if "," in text.splitlines()[0]:
        return list(csv.DictReader(io.StringIO(text)))
    lines = [line.split() for line in text.splitlines() if line.strip()]
    return [dict(zip(lines[0], values, strict=True)) for values in lines[1:]]


def label_map(rows: list[dict[str, str]], column: str) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in rows:
        patient = str(int(row.get("patient_id") or row["test_patient_id"]))
        value = int(row[column])
        if patient in labels and labels[patient] != value:
            raise ValueError("Conflicting labels for the same patient.")
        labels[patient] = value
    return labels


def counts(labels: dict[str, int]) -> dict[str, int]:
    return {str(label): count for label, count in sorted(Counter(labels.values()).items())}


def comparison(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    shared = left.keys() & right.keys()
    return {
        "shared_patients": len(shared),
        "different_labels": sum(left[patient] != right[patient] for patient in shared),
        "only_left": len(left.keys() - right.keys()),
        "only_right": len(right.keys() - left.keys()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-public", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_path = ROOT / "data/raw/brugada-huca/1.0.0/metadata.csv"
    manifest_path = ROOT / "data/brugada_huca/labels/all_labels.csv"
    raw_payload = raw_path.read_bytes()
    raw_rows = read_rows(raw_payload)
    raw_labels = label_map(raw_rows, "brugada")
    processed_rows = read_rows(manifest_path.read_bytes())
    processed_labels = label_map(processed_rows, "clinical_brugada")
    retained = [
        row
        for row in raw_rows
        if int(row["brugada"]) in (0, 1)
        and not (int(row["basal_pattern"]) == 1 and int(row["brugada"]) == 1)
    ]
    selected_labels = label_map(retained, "brugada")
    prediction_paths = sorted(
        (ROOT / "outputs/final_results_tfg/runs/cnn_clinical_brugada_loocv").glob(
            "*/resnet18/k*_seed*/fold_predictions.csv"
        )
    )
    canonical_maps = [
        label_map(read_rows(path.read_bytes()), "true_label") for path in prediction_paths
    ]
    if len(canonical_maps) != 30 or any(len(labels) != 317 for labels in canonical_maps):
        raise ValueError("Expected 30 canonical runs with 317 patients each.")
    if any(labels != canonical_maps[0] for labels in canonical_maps):
        raise ValueError("Canonical clinical runs disagree on their evaluation labels.")
    canonical_labels = canonical_maps[0]
    report: dict[str, object] = {
        "checked_on": date.today().isoformat(),
        "scope": "Metadata and labels only. No ECG signals or results were modified.",
        "local_metadata_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "local_metadata_label_counts": counts(raw_labels),
        "local_metadata_cross_counts": {
            f"basal_{basal}_brugada_{label}": count
            for (basal, label), count in sorted(
                Counter(
                    (int(row["basal_pattern"]), int(row["brugada"])) for row in raw_rows
                ).items()
            )
        },
        "current_builder_selection_label_counts": counts(selected_labels),
        "processed_manifest": {
            "patients": len(processed_labels),
            "rows": len(processed_rows),
            "leads": dict(sorted(Counter(row["lead"] for row in processed_rows).items())),
            "label_counts": counts(processed_labels),
        },
        "canonical_clinical_runs": len(canonical_maps),
        "canonical_label_counts": counts(canonical_labels),
        "builder_vs_canonical": comparison(selected_labels, canonical_labels),
        "processed_vs_canonical": comparison(processed_labels, canonical_labels),
    }
    if args.check_public:
        with urlopen(PUBLIC_ROOT + "metadata.csv", timeout=60) as response:
            public_payload = response.read()
        public_labels = label_map(read_rows(public_payload), "brugada")
        with urlopen(PUBLIC_ROOT + "SHA256SUMS.txt", timeout=60) as response:
            checksum_text = response.read().decode("utf-8")
        checksum_lines = [
            line.split()
            for line in checksum_text.splitlines()
            if line.split() and line.split()[-1].lstrip("*./") == "metadata.csv"
        ]
        public_hash = hashlib.sha256(public_payload).hexdigest()
        if len(checksum_lines) != 1 or checksum_lines[0][0].lower() != public_hash:
            raise ValueError("Public metadata do not match the publisher's SHA-256.")
        report["public_metadata"] = {
            "url": PUBLIC_ROOT + "metadata.csv",
            "sha256": public_hash,
            "publisher_checksum_verified": True,
            "label_counts": counts(public_labels),
            "public_vs_local": comparison(public_labels, raw_labels),
            "public_vs_canonical": comparison(public_labels, canonical_labels),
            "public_to_local_label_counts": dict(
                sorted(
                    Counter(
                        f"{public_labels[patient]}_to_{raw_labels[patient]}"
                        for patient in public_labels.keys() & raw_labels.keys()
                    ).items()
                )
            ),
            "public_to_canonical_label_counts": dict(
                sorted(
                    Counter(
                        f"{public_labels[patient]}_to_{canonical_labels[patient]}"
                        for patient in public_labels.keys() & canonical_labels.keys()
                    ).items()
                )
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
