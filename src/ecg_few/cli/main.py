"""Dispatch commands without loading optional backends at startup."""

from __future__ import annotations

import argparse
import sys
from importlib import import_module

COMMANDS = {
    "build-synthetic": ("ecg_few.datasets.synthetic", "Generate a synthetic ECG dataset."),
    "build-huca": ("ecg_few.datasets.huca", "Prepare locally available HUCA records."),
    "cnn": ("ecg_few.cnn.morphology.pipeline", "Run morphological CNN evaluation."),
    "cnn-clinical": ("ecg_few.cnn.clinical.pipeline", "Run clinical CNN evaluation."),
    "vlm": ("ecg_few.vlm.pipeline", "Run VLM evaluation."),
    "validate-vlm": ("ecg_few.vlm.validation", "Validate a VLM setup without inference."),
    "compare": ("ecg_few.reporting.comparison", "Compare aligned CNN and VLM predictions."),
    "compare-domains": ("ecg_few.reporting.sim_real", "Compare synthetic and clinical reports."),
    "compare-adaptation": (
        "ecg_few.reporting.domain_adaptation",
        "Compare domain adaptation reports.",
    ),
    "audit": ("ecg_few.evaluation.audit", "Audit a provisional campaign."),
    "audit-clinical": ("ecg_few.evaluation.clinical_audit", "Audit a clinical CNN campaign."),
}


def main(argv: list[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="ecg-few", description="Few-shot ECG image experiments.")
    parser.add_argument("command", choices=COMMANDS, help="Experiment or maintenance command.")
    parser.epilog = "\n".join(
        f"{name}: {description}" for name, (_, description) in COMMANDS.items()
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return
    command = parser.parse_args(arguments[:1]).command
    try:
        module = import_module(COMMANDS[command][0])
    except ModuleNotFoundError as exc:
        if exc.name in {"torch", "torchvision", "transformers", "accelerate", "wfdb"}:
            parser.error(
                f"Missing optional dependency {exc.name}. Install the corresponding extra."
            )
        raise
    module.main(arguments[1:])
