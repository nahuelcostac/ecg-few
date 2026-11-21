"""Clinical for ECG experiments."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from ecg_few.cnn.clinical.settings import BASE_SEED_OFFSET, CONDITIONS
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
    PatientRecord,
    patients_from_rows,
    selection_for,
)


def selection_for_condition(
    *,
    fold: dict[str, Any],
    rows: Sequence[BrugadaImageRow],
    test_patient_id: str,
    k: int,
    seed: int,
    condition: str,
    fold_id: int,
) -> dict[str, list[str]]:
    base = selection_for(fold, k=k, seed=seed)
    if condition == "estandar":
        return base
    patients = patients_from_rows(rows)
    validation_ids = base["validation_patient_ids"]
    context_ids = select_balanced_clinical_context_patient_ids(
        patients,
        test_patient_id=test_patient_id,
        excluded_patient_ids=validation_ids,
        k=k,
        seed=seed + fold_id * BASE_SEED_OFFSET,
    )
    return {
        "context_patient_ids": context_ids,
        "validation_patient_ids": validation_ids,
    }


def select_balanced_clinical_context_patient_ids(
    patients: Sequence[PatientRecord],
    *,
    test_patient_id: str,
    excluded_patient_ids: Sequence[str] = (),
    k: int,
    seed: int,
) -> list[str]:
    excluded = {str(test_patient_id), *(str(item) for item in excluded_patient_ids)}
    available = [patient for patient in patients if patient.patient_id not in excluded]
    rng = random.Random(seed)
    by_label = {
        label: [patient for patient in available if patient.reference_brugada == label]
        for label in (0, 1)
    }
    for candidates in by_label.values():
        candidates.sort(key=lambda patient: patient.patient_id)
        rng.shuffle(candidates)
    n_positive = k // 2
    n_negative = k - n_positive
    selected = by_label[1][:n_positive] + by_label[0][:n_negative]
    selected_ids = {patient.patient_id for patient in selected}
    if len(selected) < k:
        remaining = [patient for patient in available if patient.patient_id not in selected_ids]
        remaining.sort(key=lambda patient: patient.patient_id)
        rng.shuffle(remaining)
        selected.extend(remaining[: k - len(selected)])
    if k >= 2 and {patient.reference_brugada for patient in selected} != {0, 1}:
        raise ValueError(f"Balanced selection failed for k={k} and fold test={test_patient_id}.")
    return [patient.patient_id for patient in selected[:k]]


def assert_no_leakage(
    test_patient_id: str, context_ids: Sequence[str], validation_ids: Sequence[str]
) -> None:
    context_set = set(context_ids)
    validation_set = set(validation_ids)
    if test_patient_id in context_set or test_patient_id in validation_set:
        raise ValueError(f"Test patient leaked into train/validation: {test_patient_id}")
    if context_set.intersection(validation_set):
        raise ValueError("Training and validation patients overlap.")
    if len(context_ids) != len(context_set):
        raise ValueError("Training context contains duplicate patients.")


def validate_clinical_rows(rows: Sequence[BrugadaImageRow]) -> None:
    if not rows:
        raise ValueError("The clinical CNN dataset contains no V1 rows.")
    if any(row.clinical_brugada is None for row in rows):
        raise ValueError("Every clinical CNN row must have clinical_brugada.")
    patient_ids = {row.patient_id for row in rows}
    if len(patient_ids) != len(rows):
        raise ValueError("The clinical CNN requires exactly one V1 image per patient.")
    counts = {label: sum(int(row.clinical_brugada == label) for row in rows) for label in (0, 1)}
    if counts != {0: 201, 1: 116}:
        raise ValueError(f"Unexpected HUCA V1 clinical counts: {counts}")


def parse_conditions(text: str) -> list[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    unknown = sorted(set(values) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unsupported clinical CNN conditions: {unknown}")
    if not values:
        raise ValueError("At least one clinical CNN condition is required.")
    return list(dict.fromkeys(values))
