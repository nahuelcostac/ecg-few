"""Verify an installed wheel with synthetic inputs outside the repository."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True, type=Path)
    args = parser.parse_args()
    interpreter = args.python.absolute()
    repository = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="ecg-wheel-") as temporary:
        working = Path(temporary)

        def run(*arguments: str) -> None:
            subprocess.run([str(interpreter), "-I", *arguments], cwd=working, check=True)

        run(
            "-c",
            "import ecg_few; from ecg_few.prompts import load_markdown_prompt; "
            "assert 'site-packages' in ecg_few.__file__; "
            "assert load_markdown_prompt('package:system/qrs_huca.md'); "
            "assert load_markdown_prompt('package:system/clinical_brugada_huca.md')",
        )
        run("-m", "ecg_few", "--help")
        dataset = working / "dataset"
        output = working / "campaign"
        run(
            "-m",
            "ecg_few",
            "build-synthetic",
            "--config",
            str(repository / "configs/smoke_dataset.toml"),
            "--outdir",
            str(dataset),
        )
        experiment = (
            "-m",
            "ecg_few",
            "vlm",
            "--config",
            str(repository / "configs/smoke_vlm.toml"),
            "--dataset-root",
            str(dataset),
            "--output-root",
            str(output),
        )
        run(*experiment)
        predictions = {path: path.read_bytes() for path in output.rglob("fold_predictions.jsonl")}
        assert predictions
        run(*experiment)
        assert all(path.read_bytes() == content for path, content in predictions.items())
        assert (output / "reports/vlm_summary_by_seed.csv").exists()
        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "provisional"
    print("Installed wheel checks passed.")


if __name__ == "__main__":
    main()
