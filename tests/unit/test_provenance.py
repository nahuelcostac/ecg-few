import pytest

from ecg_few.datasets.provenance import comparison, counts, label_map, read_rows


def test_metadata_csv_and_whitespace_have_the_same_labels():
    comma = b"patient_id,brugada\n101,0\n102,2\n"
    whitespace = b"patient_id brugada\n101 0\n102 2\n"
    assert label_map(read_rows(comma), "brugada") == label_map(read_rows(whitespace), "brugada")
    assert counts(label_map(read_rows(comma), "brugada")) == {"0": 1, "2": 1}


def test_repeated_leads_do_not_count_as_new_patients():
    rows = [
        {"patient_id": "101", "clinical_brugada": "1", "lead": lead} for lead in ("V1", "V2", "V3")
    ]
    assert label_map(rows, "clinical_brugada") == {"101": 1}


def test_disagreement_within_a_patient_is_not_silently_overwritten():
    rows = [{"patient_id": "101", "brugada": str(label)} for label in (0, 1)]
    with pytest.raises(ValueError, match="Conflicting labels"):
        label_map(rows, "brugada")


def test_comparison_reports_label_changes_and_missing_patients_separately():
    summary = comparison({"101": 0, "102": 1, "103": 2}, {"101": 1, "102": 1, "104": 0})
    assert summary == {
        "shared_patients": 2,
        "different_labels": 1,
        "only_left": 1,
        "only_right": 1,
    }
    assert all(isinstance(value, int) for value in summary.values())


def test_prediction_patient_column_is_supported():
    assert label_map([{"test_patient_id": "101", "true_label": "0"}], "true_label") == {"101": 0}
