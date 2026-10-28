#!/usr/bin/env python3
"""Audit direct clinical CNN LOOCV outputs without exposing patient-level data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit clinical CNN LOOCV outputs.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/work/cnn_clinical_brugada_loocv"),
    )
    parser.add_argument("--expected-folds", type=int, default=317)
    parser.add_argument("--k-values", default="2,4,8,16,32")
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--conditions", default="estandar,balanced")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = parse_int_list(args.k_values)
    seeds = parse_int_list(args.seeds)
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    runs_root = args.output_root / "runs"
    if not runs_root.exists() and all((args.output_root / value).exists() for value in conditions):
        runs_root = args.output_root
    run_count = 0
    for condition in conditions:
        for k in k_values:
            for seed in seeds:
                path = (
                    runs_root
                    / condition
                    / "resnet18"
                    / f"k{k}_seed{seed}"
                    / "fold_predictions.jsonl"
                )
                records = read_jsonl(path)
                audit_run(records, path, args.expected_folds, condition, k, seed)
                run_count += 1
    print(f"[OK] Audited {run_count} clinical CNN runs with no detected leakage.")


def audit_run(
    records: list[dict[str, object]],
    path: Path,
    expected_folds: int,
    condition: str,
    k: int,
    seed: int,
) -> None:
    if len(records) != expected_folds:
        raise ValueError(
            f"{path} has {len(records)} predictions, expected {expected_folds} "
            f"for condition={condition}, k={k}, seed={seed}."
        )
    fold_ids = [int(record["fold_id"]) for record in records]
    if sorted(fold_ids) != list(range(expected_folds)):
        raise ValueError(f"{path} does not contain every LOOCV fold exactly once.")
    keys = [str(record["key"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path} contains duplicate prediction keys.")
    for record in records:
        test_patient = str(record["patient_id"])
        context_ids = split_ids(record.get("context_patient_ids", ""))
        validation_ids = split_ids(record.get("validation_patient_ids", ""))
        if test_patient in context_ids or test_patient in validation_ids:
            raise ValueError(f"{path} leaks the test patient.")
        if set(context_ids).intersection(validation_ids):
            raise ValueError(f"{path} overlaps context and validation patients.")
        if len(context_ids) != k:
            raise ValueError(f"{path} has an invalid context size.")
        for key in (
            "true_label",
            "pred_label",
            "prob_clinical_brugada",
            "threshold",
        ):
            if key not in record:
                raise ValueError(f"{path} is missing {key}.")


def split_ids(value: object) -> list[str]:
    return [item for item in str(value).split("|") if item]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
