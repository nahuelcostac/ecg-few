from ecg_few.evaluation.audit import audit_vlm_readiness


def test_readiness_uses_installed_modules_and_prompts_outside_repository(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    labels = dataset / "labels"
    labels.mkdir(parents=True)
    (labels / "all_labels.csv").touch()
    (labels / "loocv_folds.jsonl").touch()
    monkeypatch.chdir(tmp_path)

    assert audit_vlm_readiness(dataset) == {"complete": True, "errors": []}


def test_readiness_still_reports_missing_dataset_files(tmp_path):
    result = audit_vlm_readiness(tmp_path)

    assert result["complete"] is False
    assert len(result["errors"]) == 2
    assert any("all_labels.csv" in error for error in result["errors"])
    assert any("loocv_folds.jsonl" in error for error in result["errors"])
