"""Thresholds for ECG experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ecg_few.evaluation.metrics import brugada_metrics
from ecg_few.findings import LABEL_NAMES, findings_to_brugada
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
)


def select_patient_threshold(
    *,
    rows: Sequence[BrugadaImageRow],
    y_prob: np.ndarray,
    strategy: str,
    default_threshold: float,
) -> float:
    if strategy == "fixed" or y_prob.size == 0:
        return default_threshold
    true_binary = patient_reference_labels(rows)
    if np.unique(true_binary).size < 2:
        return default_threshold
    candidates = np.unique(
        np.concatenate(
            [
                y_prob.reshape(-1).astype(np.float64),
                np.linspace(0.05, 0.95, 19, dtype=np.float64),
                np.asarray([default_threshold], dtype=np.float64),
            ]
        )
    )
    best_threshold = default_threshold
    best_score = -math.inf
    metric_key = "f1" if strategy == "val_derived_f1" else "balanced_accuracy"
    for candidate in candidates:
        pred_binary = patient_predictions_at_threshold(rows, y_prob, float(candidate))
        score = brugada_metrics(true_binary, pred_binary).get(metric_key)
        if score is None:
            continue
        if float(score) > best_score:
            best_score = float(score)
            best_threshold = float(candidate)
    return best_threshold


def score_patient_threshold(
    *,
    rows: Sequence[BrugadaImageRow],
    y_prob: np.ndarray,
    threshold: float,
    metric: str,
) -> float:
    true_binary = patient_reference_labels(rows)
    pred_binary = patient_predictions_at_threshold(rows, y_prob, threshold)
    score = brugada_metrics(true_binary, pred_binary).get(metric)
    return float(score or 0.0)


def patient_reference_labels(rows: Sequence[BrugadaImageRow]) -> np.ndarray:
    by_patient: dict[str, int] = {}
    for row in rows:
        by_patient.setdefault(row.patient_id, int(row.reference_brugada))
    return np.asarray([by_patient[patient_id] for patient_id in sorted(by_patient)], dtype=np.int64)


def patient_predictions_at_threshold(
    rows: Sequence[BrugadaImageRow],
    y_prob: np.ndarray,
    threshold: float,
) -> np.ndarray:
    predictions: dict[str, int] = {}
    for patient_id, probs in patient_probability_matrix(rows, y_prob).items():
        findings = {
            label_name: bool(probs[index] >= threshold)
            for index, label_name in enumerate(LABEL_NAMES)
        }
        predictions[patient_id] = findings_to_brugada(findings)
    return np.asarray(
        [predictions[patient_id] for patient_id in sorted(predictions)],
        dtype=np.int64,
    )


def patient_probability_matrix(
    rows: Sequence[BrugadaImageRow],
    y_prob: np.ndarray,
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for row, probs in zip(rows, y_prob, strict=True):
        grouped.setdefault(row.patient_id, []).append(np.asarray(probs, dtype=np.float64))
    return {
        patient_id: np.mean(np.vstack(values), axis=0) for patient_id, values in grouped.items()
    }


def patient_condition_probabilities(
    rows: Sequence[BrugadaImageRow],
    y_prob: np.ndarray,
) -> dict[str, float]:
    matrix = patient_probability_matrix(rows, y_prob)
    if len(matrix) != 1:
        raise ValueError("Expected rows for exactly one patient.")
    probs = next(iter(matrix.values()))
    return {label_name: float(probs[index]) for index, label_name in enumerate(LABEL_NAMES)}


def lead_probabilities(
    rows: Sequence[BrugadaImageRow],
    y_prob: np.ndarray,
) -> dict[str, dict[str, float]]:
    return {
        row.lead: {label_name: float(probs[index]) for index, label_name in enumerate(LABEL_NAMES)}
        for row, probs in zip(rows, y_prob, strict=True)
    }
