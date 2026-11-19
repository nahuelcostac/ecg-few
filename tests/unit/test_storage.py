import json
from pathlib import Path

import pytest

from ecg_few.datasets.storage import reset_output_dir
from ecg_few.experiment_io import prepare_campaign, save_json, validate_output_path
from ecg_few.experiment_io.safety import public_config


@pytest.mark.parametrize(
    "destination",
    [
        "outputs/final_results_tfg/new.json",
        "reports/loocv/cnn/new.json",
        "data/raw/brugada-huca/new.json",
        "outputs/work/../final_results_tfg/new.json",
    ],
)
def test_protected_destinations_reject_writes(tmp_path, monkeypatch, destination):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Protected"):
        save_json(Path(destination), {"value": 1})
    assert not (tmp_path / destination).exists()


def test_dataset_overwrite_preserves_layout(tmp_path):
    root = tmp_path / "dataset"
    images = root / "images"
    images.mkdir(parents=True)
    marker = images / ".gitkeep"
    marker.touch()
    readme = root / "README.md"
    readme.write_text("Dataset layout.\n", encoding="utf-8")
    image = images / "generated.png"
    image.write_bytes(b"synthetic")
    with pytest.raises(FileExistsError):
        reset_output_dir(root, overwrite=False)
    assert image.exists()
    reset_output_dir(root, overwrite=True)
    assert marker.exists()
    assert readme.read_text(encoding="utf-8") == "Dataset layout.\n"
    assert not image.exists()


def test_dataset_reset_rejects_unrelated_files(tmp_path):
    source = tmp_path / "important.txt"
    source.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        reset_output_dir(tmp_path, overwrite=True)
    assert source.read_text(encoding="utf-8") == "keep"


def test_campaign_resume_checks_inputs_configuration_and_prompts(tmp_path):
    source = tmp_path / "input.txt"
    source.write_text("first", encoding="utf-8")
    output = tmp_path / "campaign"
    arguments = {"max_output_tokens": 32, "api_key": "private"}
    kwargs = {"inputs": {"input": source}, "resources": {"prompt": "first"}}
    prepare_campaign(output, arguments, resume=False, **kwargs)
    manifest = output / "run_manifest.json"
    original = manifest.read_bytes()
    assert "private" not in original.decode()
    assert json.loads(original)["config"]["max_output_tokens"] == 32
    prepare_campaign(output, arguments, resume=True, **kwargs)
    assert manifest.read_bytes() == original
    with pytest.raises(ValueError, match="changed"):
        prepare_campaign(output, {**arguments, "max_output_tokens": 64}, resume=True, **kwargs)
    with pytest.raises(ValueError, match="changed"):
        prepare_campaign(
            output, arguments, resume=True, inputs=kwargs["inputs"], resources={"prompt": "second"}
        )
    source.write_text("second", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        prepare_campaign(output, arguments, resume=True, **kwargs)


def test_campaign_rejects_untracked_existing_predictions(tmp_path):
    (tmp_path / "fold_predictions.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        prepare_campaign(tmp_path, {}, inputs={}, resume=True)


def test_resume_rejects_changed_image_contents(tmp_path):
    dataset = tmp_path / "dataset"
    labels = dataset / "labels"
    labels.mkdir(parents=True)
    image = dataset / "image.png"
    image.write_bytes(b"first")
    manifest = labels / "all_labels.csv"
    manifest.write_text("image_path\nimage.png\n", encoding="utf-8")
    output = tmp_path / "campaign"
    prepare_campaign(output, {}, inputs={"dataset": manifest}, resume=False)
    image.write_bytes(b"second")
    with pytest.raises(ValueError, match="changed"):
        prepare_campaign(output, {}, inputs={"dataset": manifest}, resume=True)


def test_public_configuration_removes_credentials_recursively():
    result = public_config(
        {"max_output_tokens": 12, "backend": {"hf_token": "private", "model": "test"}}
    )
    assert result == {"max_output_tokens": 12, "backend": {"model": "test"}}


def test_provisional_output_is_allowed(tmp_path):
    assert validate_output_path(tmp_path / "outputs/work/test") == tmp_path / "outputs/work/test"
