"""Selection for ECG experiments."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from ecg_few.protocols.loocv import (
    BrugadaImageRow,
    patients_from_rows,
    select_context_patient_ids,
    selection_for,
)
from ecg_few.vlm.messages import patient_sort_key
from ecg_few.vlm.runtime import (
    resolve_model_name,
)
from ecg_few.vlm.settings import (
    BALANCED_CONDITION,
    CLINICAL_TASK,
    DEFAULT_CLINICAL_PROMPT,
    DEFAULT_CLINICAL_SYSTEM_PROMPT,
    DEFAULT_MORPHOLOGY_PROMPT,
    DEFAULT_MORPHOLOGY_SYSTEM_PROMPT,
    NO_SUPPORT_IMAGES_CONDITION,
    ONLY_LEAD,
    PERMUTED_CONDITION,
    ZERO_SHOT_CONDITION,
)


def resolve_models(args: argparse.Namespace, runtime: str) -> list[str]:
    model_list = parse_string_list(str(getattr(args, "models", "") or ""))
    if model_list:
        return model_list
    return [resolve_model_name(str(getattr(args, "model", "")), runtime)]


def parse_string_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def only_v1_rows(rows: list[BrugadaImageRow]) -> list[BrugadaImageRow]:
    return [row for row in rows if row.lead.upper() == ONLY_LEAD]


def resolve_context_dataset(
    dataset_root: Path,
    context_dataset_root: Path | str,
) -> tuple[Path, bool]:
    text = str(context_dataset_root)
    if text in {"", "."}:
        return dataset_root, False
    resolved_context_root = Path(context_dataset_root).resolve()
    if resolved_context_root == dataset_root:
        return dataset_root, False
    return resolved_context_root, True


def resolve_clinical_leads(args: argparse.Namespace) -> list[str]:
    leads = [lead.upper() for lead in parse_string_list(str(getattr(args, "clinical_leads", "")))]
    if not leads:
        leads = [str(getattr(args, "clinical_lead", "V1")).upper()]
    seen: set[str] = set()
    unique_leads: list[str] = []
    for lead in leads:
        if lead in seen:
            continue
        seen.add(lead)
        unique_leads.append(lead)
    if not unique_leads:
        raise ValueError("TASK=clinical requires at least one clinical lead.")
    if unique_leads != [ONLY_LEAD]:
        raise ValueError("This pipeline is configured to use V1 only.")
    return unique_leads


def resolve_prompt_path(path: str, *, task: str, kind: str) -> str:
    if path:
        return path
    if task == CLINICAL_TASK:
        return DEFAULT_CLINICAL_SYSTEM_PROMPT if kind == "system" else DEFAULT_CLINICAL_PROMPT
    return DEFAULT_MORPHOLOGY_SYSTEM_PROMPT if kind == "system" else DEFAULT_MORPHOLOGY_PROMPT


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")


def k_values_for_condition(
    condition: str,
    *,
    k_values: list[int],
    control_k_values: list[int],
) -> list[int]:
    if condition == ZERO_SHOT_CONDITION:
        return [0]
    positive = [k for k in k_values if k > 0]
    if condition in {PERMUTED_CONDITION, NO_SUPPORT_IMAGES_CONDITION}:
        controls = [k for k in control_k_values if k > 0]
        return [k for k in positive if k in set(controls)]
    return positive


def selection_for_condition(
    *,
    fold: dict[str, Any],
    rows: list[BrugadaImageRow],
    context_pool_rows: list[BrugadaImageRow],
    test_patient_id: str,
    k: int,
    seed: int,
    fold_id: int,
    task: str,
    condition: str,
) -> dict[str, list[str]]:
    if k == 0 or condition == ZERO_SHOT_CONDITION:
        return {"context_patient_ids": [], "validation_patient_ids": []}
    if task == CLINICAL_TASK:
        validation_ids = []
        if context_pool_rows is rows:
            validation_ids = selection_for(fold, k=k, seed=seed)["validation_patient_ids"]
        if condition == BALANCED_CONDITION:
            return {
                "context_patient_ids": select_balanced_clinical_context_patient_ids(
                    context_pool_rows,
                    test_patient_id=test_patient_id,
                    k=k,
                    seed=seed + fold_id * 100_003,
                ),
                "validation_patient_ids": validation_ids,
            }
        if context_pool_rows is rows:
            return {
                "context_patient_ids": selection_for(fold, k=k, seed=seed)["context_patient_ids"],
                "validation_patient_ids": validation_ids,
            }
        patients = patients_from_rows(context_pool_rows)
        return {
            "context_patient_ids": select_context_patient_ids(
                patients,
                test_patient_id="__real_eval_patient__",
                k=k,
                seed=seed + fold_id * 100_003,
            ),
            "validation_patient_ids": validation_ids,
        }
    if context_pool_rows is rows and condition == BALANCED_CONDITION:
        return {
            "context_patient_ids": select_balanced_context_patient_ids(
                rows,
                test_patient_id=test_patient_id,
                k=k,
                seed=seed + fold_id * 100_003,
            ),
            "validation_patient_ids": selection_for(fold, k=k, seed=seed)["validation_patient_ids"],
        }
    return selection_for(fold, k=k, seed=seed)


def answer_rows_for_condition(
    context_rows: list[BrugadaImageRow],
    *,
    condition: str,
    seed: int,
) -> list[BrugadaImageRow]:
    if condition != PERMUTED_CONDITION or len(context_rows) < 2:
        return context_rows
    shuffled = list(context_rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    unchanged = all(
        left.patient_id == right.patient_id and left.lead == right.lead
        for left, right in zip(context_rows, shuffled, strict=True)
    )
    if unchanged:
        shuffled = shuffled[1:] + shuffled[:1]
    return shuffled


def select_qrs_context_patient_ids(
    rows: list[BrugadaImageRow],
    *,
    k: int,
    seed: int,
) -> list[str]:
    qrs_rows = [row for row in rows if row.has_qrs_labels]
    if not qrs_rows:
        raise ValueError("Context dataset must contain explicit QRS finding labels.")
    patients = patients_from_rows(qrs_rows)
    return select_context_patient_ids(
        patients,
        test_patient_id="__real_eval_patient__",
        k=k,
        seed=seed,
    )


def select_balanced_context_patient_ids(
    rows: list[BrugadaImageRow],
    *,
    test_patient_id: str,
    k: int,
    seed: int,
) -> list[str]:
    qrs_rows = [row for row in rows if row.has_qrs_labels]
    if not qrs_rows:
        raise ValueError("Context dataset must contain explicit QRS finding labels.")
    patients = [
        patient
        for patient in patients_from_rows(qrs_rows)
        if patient.patient_id != str(test_patient_id)
    ]
    rng = random.Random(seed)
    by_label = {
        label: [patient for patient in patients if patient.reference_brugada == label]
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
        remaining = [patient for patient in patients if patient.patient_id not in selected_ids]
        remaining.sort(key=lambda patient: patient.patient_id)
        rng.shuffle(remaining)
        selected.extend(remaining[: k - len(selected)])
    selected = selected[:k]
    return [patient.patient_id for patient in selected]


def select_balanced_clinical_context_patient_ids(
    rows: list[BrugadaImageRow],
    *,
    test_patient_id: str,
    k: int,
    seed: int,
) -> list[str]:
    patients = [
        patient
        for patient in patients_from_rows(rows)
        if patient.patient_id != str(test_patient_id)
    ]
    rng = random.Random(seed)
    by_label = {
        label: [patient for patient in patients if patient.reference_brugada == label]
        for label in (0, 1)
    }
    for candidates in by_label.values():
        candidates.sort(key=lambda patient: patient_sort_key(patient.patient_id))
        rng.shuffle(candidates)
    n_positive = k // 2
    n_negative = k - n_positive
    selected = by_label[1][:n_positive] + by_label[0][:n_negative]
    selected_ids = {patient.patient_id for patient in selected}
    if len(selected) < k:
        remaining = [patient for patient in patients if patient.patient_id not in selected_ids]
        remaining.sort(key=lambda patient: patient_sort_key(patient.patient_id))
        rng.shuffle(remaining)
        selected.extend(remaining[: k - len(selected)])
    if k >= 2 and by_label[0] and by_label[1]:
        labels = {patient.reference_brugada for patient in selected}
        if labels != {0, 1}:
            raise ValueError(
                f"Could not select both normal and Brugada real-context examples for k={k}."
            )
    selected = selected[:k]
    return [patient.patient_id for patient in selected]
