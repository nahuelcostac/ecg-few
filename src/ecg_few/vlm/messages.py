"""Messages for ECG experiments."""

from __future__ import annotations

import base64
import json
import random
from pathlib import Path
from typing import Any

from ecg_few.prompts import (
    multilabel_answer_text,
)
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
)
from ecg_few.vlm.runtime import (
    image_for_local,
)
from ecg_few.vlm.settings import CLINICAL_LABEL_NAME, PERMUTED_CONDITION


def build_clinical_messages(
    *,
    dataset_root: Path,
    context_dataset_root: Path,
    system_prompt: str,
    prompt_template: str,
    context_rows: list[BrugadaImageRow],
    condition: str,
    seed: int,
    test_row: BrugadaImageRow,
    clinical_leads: list[str],
    include_support_images: bool,
    local: bool,
) -> list[dict[str, Any]]:
    system_content: str | list[dict[str, str]]
    system_content = [{"type": "text", "text": system_prompt}] if local else system_prompt
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
    context_by_patient = rows_by_patient(context_rows)
    answers_by_patient = clinical_answer_labels_for_condition(
        context_rows,
        condition=condition,
        seed=seed,
    )
    for patient_id in sorted(context_by_patient, key=patient_sort_key):
        for support_row in rows_for_leads(context_by_patient[patient_id], clinical_leads):
            messages.append(
                support_patient_message(
                    prompt_for_row(prompt_template, support_row),
                    context_dataset_root / support_row.image_path,
                    include_images=include_support_images,
                    local=local,
                )
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": clinical_answer_text(answers_by_patient[patient_id]),
                }
            )
    messages.append(
        patient_message(
            prompt_for_row(prompt_template, test_row),
            dataset_root / test_row.image_path,
            local=local,
        )
    )
    return messages


def build_messages(
    *,
    dataset_root: Path,
    context_dataset_root: Path,
    system_prompt: str,
    prompt_template: str,
    context_rows: list[BrugadaImageRow],
    answer_rows: list[BrugadaImageRow],
    test_row: BrugadaImageRow,
    include_support_images: bool,
    local: bool,
) -> list[dict[str, Any]]:
    system_content: str | list[dict[str, str]]
    system_content = [{"type": "text", "text": system_prompt}] if local else system_prompt
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_content}]
    for row, answer_row in zip(context_rows, answer_rows, strict=True):
        messages.append(
            support_user_message(
                prompt_for_row(prompt_template, row),
                context_dataset_root / row.image_path,
                include_image=include_support_images,
                local=local,
            )
        )
        messages.append(
            {
                "role": "assistant",
                "content": multilabel_answer_text(answer_row.expected_answer()),
            }
        )
    messages.append(
        user_message(
            prompt_for_row(prompt_template, test_row),
            dataset_root / test_row.image_path,
            local=local,
        )
    )
    return messages


def prompt_for_row(template: str, row: BrugadaImageRow) -> str:
    return template.replace("{lead}", row.lead)


def patient_sort_key(patient_id: str) -> tuple[int, str]:
    return (int(patient_id), patient_id) if str(patient_id).isdigit() else (10**9, patient_id)


def sorted_rows_by_lead(rows: list[BrugadaImageRow]) -> list[BrugadaImageRow]:
    order = {"V1": 0}
    return sorted(rows, key=lambda row: (order.get(row.lead.upper(), 99), row.lead))


def row_for_lead(rows: list[BrugadaImageRow], lead: str) -> BrugadaImageRow:
    requested = lead.upper()
    for row in sorted_rows_by_lead(rows):
        if row.lead.upper() == requested:
            return row
    available = ", ".join(row.lead for row in sorted_rows_by_lead(rows))
    patient_id = rows[0].patient_id if rows else "?"
    raise ValueError(f"Patient {patient_id} has no {requested}; available: {available}")


def rows_for_leads(rows: list[BrugadaImageRow], leads: list[str]) -> list[BrugadaImageRow]:
    return [row_for_lead(rows, lead) for lead in leads]


def rows_by_patient(rows: list[BrugadaImageRow]) -> dict[str, list[BrugadaImageRow]]:
    grouped: dict[str, list[BrugadaImageRow]] = {}
    for row in rows:
        grouped.setdefault(row.patient_id, []).append(row)
    return {
        patient_id: sorted_rows_by_lead(patient_rows)
        for patient_id, patient_rows in grouped.items()
    }


def clinical_answers_by_patient(rows: list[BrugadaImageRow]) -> dict[str, bool]:
    return {
        patient_id: bool(patient_rows[0].reference_brugada)
        for patient_id, patient_rows in rows_by_patient(rows).items()
    }


def clinical_answer_labels_for_condition(
    rows: list[BrugadaImageRow],
    *,
    condition: str,
    seed: int,
) -> dict[str, bool]:
    answers = clinical_answers_by_patient(rows)
    if condition != PERMUTED_CONDITION or len(answers) < 2:
        return answers
    patient_ids = sorted(answers, key=patient_sort_key)
    labels = [answers[patient_id] for patient_id in patient_ids]
    rng = random.Random(seed)
    rng.shuffle(labels)
    unchanged = all(
        answers[patient_id] == label for patient_id, label in zip(patient_ids, labels, strict=True)
    )
    if unchanged:
        labels = labels[1:] + labels[:1]
    return {patient_id: label for patient_id, label in zip(patient_ids, labels, strict=True)}


def clinical_answer_text(value: bool) -> str:
    return json.dumps({CLINICAL_LABEL_NAME: bool(value)}, sort_keys=True)


def support_user_message(
    prompt: str,
    image_path: Path,
    *,
    include_image: bool,
    local: bool,
) -> dict[str, Any]:
    if include_image:
        return user_message(prompt, image_path, local=local)
    control_prompt = (
        f"{prompt}\n\n"
        "Control condition: the support image for this demonstration is intentionally "
        "omitted. Use only the label response that follows as context."
    )
    if local:
        return {"role": "user", "content": [{"type": "text", "text": control_prompt}]}
    return {"role": "user", "content": [{"type": "text", "text": control_prompt}]}


def support_patient_message(
    prompt: str,
    image_path: Path,
    *,
    include_images: bool,
    local: bool,
) -> dict[str, Any]:
    if include_images:
        return patient_message(prompt, image_path, local=local)
    control_prompt = (
        f"{prompt}\n\n"
        "Control condition: the support image for this patient demonstration is "
        "intentionally omitted. Use only the clinical label response that follows "
        "as context."
    )
    return {"role": "user", "content": [{"type": "text", "text": control_prompt}]}


def user_message(prompt: str, image_path: Path, *, local: bool) -> dict[str, Any]:
    if local:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_for_local(image_path)},
            ],
        }
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url_for_image(image_path)}},
        ],
    }


def patient_message(prompt: str, image_path: Path, *, local: bool) -> dict[str, Any]:
    if local:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_for_local(image_path)},
            ],
        }
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url_for_image(image_path)}},
        ],
    }


def data_url_for_image(path: Path) -> str:
    mime_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(path.suffix.lower())
    if mime_type is None:
        raise ValueError(f"Unsupported image type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
