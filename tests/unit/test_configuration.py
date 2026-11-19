import argparse
from pathlib import Path

import pytest

from ecg_few.configuration import parse_config_args
from ecg_few.prompts import load_markdown_prompt


def parser():
    result = argparse.ArgumentParser()
    result.add_argument("--epochs", type=int, default=20)
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--seeds", default="42")
    result.add_argument("--output-root", type=Path)
    result.add_argument("--resume", action="store_true")
    return result


def test_toml_defaults_preserve_cli_overrides(tmp_path):
    config = tmp_path / "experiment.toml"
    config.write_text(
        '[experiment]\nepochs = 2\nseeds = [42, 123]\noutput_root = "runs/test"\nresume = true\n',
        encoding="utf-8",
    )
    args = parse_config_args(parser(), ["--config", str(config), "--epochs", "5"])
    assert args.epochs == 5
    assert args.seeds == "42,123"
    assert args.output_root == Path("runs/test")
    assert args.resume is True


@pytest.mark.parametrize("body", ["unknown = 1", 'device = "other"', 'resume = "yes"'])
def test_invalid_configuration_is_rejected(tmp_path, body):
    config = tmp_path / "invalid.toml"
    config.write_text("[experiment]\n" + body, encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_config_args(parser(), ["--config", str(config)])


def test_packaged_prompts_load_outside_repository(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for prompt in (
        "system/qrs_huca.md",
        "system/clinical_brugada_huca.md",
        "qrs/right_precordial_morphology.md",
        "clinical/brugada_patient.md",
    ):
        assert load_markdown_prompt("package:" + prompt)
        assert load_markdown_prompt("prompts/" + prompt)


def test_packaged_prompt_rejects_parent_traversal():
    with pytest.raises(ValueError, match="Invalid"):
        load_markdown_prompt("package:../private.md")
