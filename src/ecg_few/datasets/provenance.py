"""Compare label sources without exporting patient identifiers."""

from __future__ import annotations

import csv
import io
from collections import Counter


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
