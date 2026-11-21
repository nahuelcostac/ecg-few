from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("torchvision")

from ecg_few.cnn.clinical.pipeline import (
    aggregate_by_k,
    assert_no_leakage,
    build_binary_model,
    select_balanced_clinical_context_patient_ids,
)
from ecg_few.protocols.loocv import PatientRecord


def test_binary_model_has_one_clinical_output() -> None:
    model = build_binary_model("none")

    assert model.fc.out_features == 1


def test_balanced_selection_contains_both_classes_and_excludes_test() -> None:
    patients = [
        PatientRecord(
            patient_id=str(index),
            reference_brugada=index % 2,
            source_family="huca",
            basal_pattern=0,
            sudden_death=0,
            findings={},
        )
        for index in range(8)
    ]

    selected = select_balanced_clinical_context_patient_ids(
        patients,
        test_patient_id="0",
        k=4,
        seed=42,
    )

    labels = {patient.reference_brugada for patient in patients if patient.patient_id in selected}
    assert len(selected) == 4
    assert labels == {0, 1}
    assert "0" not in selected


def test_balanced_selection_excludes_validation_patients() -> None:
    patients = [
        PatientRecord(
            patient_id=str(index),
            reference_brugada=index % 2,
            source_family="huca",
            basal_pattern=0,
            sudden_death=0,
            findings={},
        )
        for index in range(8)
    ]

    selected = select_balanced_clinical_context_patient_ids(
        patients,
        test_patient_id="0",
        excluded_patient_ids=("1", "2"),
        k=4,
        seed=42,
    )

    assert not {"0", "1", "2"}.intersection(selected)


def test_leakage_check_rejects_test_or_train_validation_overlap() -> None:
    assert_no_leakage("test", ["train"], ["validation"])

    try:
        assert_no_leakage("test", ["test"], [])
    except ValueError as exc:
        assert "leaked" in str(exc)
    else:
        raise AssertionError("Test-patient leakage was not rejected.")


def test_aggregate_by_k_preserves_seed_variation() -> None:
    rows = [
        {
            "model_family": "cnn_clinical",
            "model": "resnet18",
            "task": "clinical_brugada",
            "condition": "estandar",
            "k": 2,
            "seed": seed,
            "n_patients": 317,
            "accuracy": value,
            "balanced_accuracy": value,
            "f1": value,
            "sensitivity": value,
            "specificity": value,
            "precision": value,
            "roc_auc": value,
            "average_precision": value,
            "tp": 10,
            "tn": 20,
            "fp": 5,
            "fn": 4,
        }
        for seed, value in ((42, 0.4), (123, 0.6))
    ]

    result = aggregate_by_k(rows)

    assert len(result) == 1
    assert result[0]["balanced_accuracy_mean"] == 0.5
    assert result[0]["balanced_accuracy_std"] > 0
