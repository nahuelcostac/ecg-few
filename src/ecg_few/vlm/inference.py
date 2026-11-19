"""Inference for ECG experiments."""

from __future__ import annotations

import argparse
import json
from typing import Any

from openai import OpenAI

from ecg_few.findings import LABEL_NAMES
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
)
from ecg_few.vlm.runtime import (
    LOCAL_RUNTIME,
    LocalGPUGenerator,
)
from ecg_few.vlm.settings import CLINICAL_LABEL_NAME, CLINICAL_TASK


def call_model(
    messages: list[dict[str, Any]],
    *,
    runtime: str,
    model: str,
    api_base: str | None,
    api_key: str,
    task: str,
    args: argparse.Namespace,
    generator: LocalGPUGenerator | None,
) -> str:
    if runtime == LOCAL_RUNTIME:
        if generator is None:
            raise RuntimeError("Local runtime was not initialized.")
        result = generator.generate_adaptive(
            [messages],
            initial_batch_size=int(args.local_batch_size),
            max_new_tokens=int(args.max_output_tokens),
            temperature=float(args.temperature),
        )
        return result[0].texts[0]
    client = OpenAI(
        api_key=api_key or "EMPTY",
        base_url=str(api_base).rstrip("/") + "/",
        timeout=float(args.timeout),
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(args.max_output_tokens),
    }
    if float(args.temperature) != 0:
        payload["temperature"] = float(args.temperature)
    if args.response_format == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "clinical_brugada" if task == CLINICAL_TASK else "qrs_findings",
                "schema": response_schema_for_task(task),
                "strict": True,
            },
        }
    elif args.response_format == "json_object":
        payload["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**payload)
    content = response.choices[0].message.content or ""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
    return str(content).strip()


def dry_run_prediction(row: BrugadaImageRow, *, mode: str) -> dict[str, bool]:
    if mode == "expected":
        if not row.has_qrs_labels:
            raise ValueError("Dry-run expected mode requires QRS labels.")
        return row.expected_answer()
    if mode == "negative":
        return {label_name: False for label_name in LABEL_NAMES}
    raise ValueError(f"Unsupported dry-run mode: {mode}")


def dry_run_clinical_prediction(row: BrugadaImageRow, *, mode: str) -> dict[str, bool]:
    if mode == "expected":
        return {CLINICAL_LABEL_NAME: bool(row.reference_brugada)}
    if mode == "negative":
        return {CLINICAL_LABEL_NAME: False}
    raise ValueError(f"Unsupported dry-run mode: {mode}")


def response_schema_for_task(task: str) -> dict[str, Any]:
    if task == CLINICAL_TASK:
        return {
            "type": "object",
            "properties": {CLINICAL_LABEL_NAME: {"type": "boolean"}},
            "required": [CLINICAL_LABEL_NAME],
            "additionalProperties": False,
        }
    return {
        "type": "object",
        "properties": {label_name: {"type": "boolean"} for label_name in LABEL_NAMES},
        "required": list(LABEL_NAMES),
        "additionalProperties": False,
    }


def parse_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    for index, char in enumerate(stripped):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("No valid JSON object found.")


def parse_morphology_prediction(text: str) -> dict[str, bool]:
    payload = parse_json_object(text)
    if all(type(payload.get(label)) is bool for label in LABEL_NAMES):
        return {label_name: bool(payload[label_name]) for label_name in LABEL_NAMES}
    raise ValueError("No valid QRS finding JSON prediction found.")


def parse_clinical_prediction(text: str) -> dict[str, bool]:
    payload = parse_json_object(text)
    if type(payload.get(CLINICAL_LABEL_NAME)) is bool:
        return {CLINICAL_LABEL_NAME: bool(payload[CLINICAL_LABEL_NAME])}
    raise ValueError("No valid clinical_brugada JSON prediction found.")


def aggregate_lead_findings(
    lead_predictions: dict[str, dict[str, bool]],
) -> dict[str, bool]:
    if not lead_predictions:
        return {label_name: False for label_name in LABEL_NAMES}
    return {
        label_name: (
            sum(1 for prediction in lead_predictions.values() if prediction[label_name])
            / len(lead_predictions)
            >= 0.5
        )
        for label_name in LABEL_NAMES
    }


def aggregate_clinical_lead_predictions(
    lead_predictions: dict[str, bool],
    *,
    method: str,
) -> bool:
    if not lead_predictions:
        return False
    positives = sum(1 for value in lead_predictions.values() if value)
    total = len(lead_predictions)
    if method == "majority":
        return positives / total >= 0.5
    if method == "any_positive":
        return positives > 0
    if method == "all_positive":
        return positives == total
    raise ValueError(f"Unsupported clinical aggregation method: {method}")
