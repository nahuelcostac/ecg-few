"""Default experiment settings."""

from pathlib import Path

from ecg_few.protocols.loocv import DEFAULT_K_VALUES, DEFAULT_SEEDS

DEFAULT_OUTPUT_ROOT = Path("outputs/work/cnn_clinical_brugada_loocv")
DEFAULT_K_VALUES_TEXT = ",".join(str(value) for value in DEFAULT_K_VALUES)
DEFAULT_SEEDS_TEXT = ",".join(str(value) for value in DEFAULT_SEEDS)
CONDITIONS = ("estandar", "balanced")
BASE_SEED_OFFSET = 100_003
