"""Clinical thresholds for ECG experiments."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from ecg_few.evaluation.metrics import brugada_metrics
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
)


def select_threshold(
    rows: Sequence[BrugadaImageRow],
    probabilities: np.ndarray,
    *,
    strategy: str,
    default_threshold: float,
) -> float:
    if (
        strategy == "fixed"
        or not rows
        or np.unique([int(row.clinical_brugada) for row in rows]).size < 2
    ):
        return default_threshold
    candidates = np.unique(
        np.concatenate(
            [
                probabilities.reshape(-1),
                np.linspace(0.05, 0.95, 19),
                np.asarray([default_threshold]),
            ]
        )
    )
    metric = "f1" if strategy == "val_derived_f1" else "balanced_accuracy"
    best_threshold = default_threshold
    best_score = -math.inf
    for candidate in candidates:
        score = score_threshold(rows, probabilities, float(candidate), metric=metric)
        if score > best_score:
            best_score = score
            best_threshold = float(candidate)
    return best_threshold


def score_threshold(
    rows: Sequence[BrugadaImageRow],
    probabilities: np.ndarray,
    threshold: float,
    *,
    metric: str,
) -> float:
    y_true = [int(row.clinical_brugada) for row in rows]
    y_pred = [int(value >= threshold) for value in probabilities]
    result = brugada_metrics(y_true, y_pred).get(metric)
    return 0.0 if result is None else float(result)
