#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
UV="${UV:-uv}"

DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/brugada_huca}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/work/cnn_clinical_brugada_loocv}"
K_VALUES="${K_VALUES:-2,4,8,16,32}"
SEEDS="${SEEDS:-42,123,2026}"
CONDITIONS="${CONDITIONS:-estandar,balanced}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-auto}"
LIMIT_FOLDS="${LIMIT_FOLDS:-0}"
RESUME="${RESUME:-1}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH MPLBACKEND

run_python() {
  if command -v "$UV" >/dev/null 2>&1; then
    "$UV" run --no-sync python "$@"
  else
    python "$@"
  fi
}

ARGS=""
if [ "$LIMIT_FOLDS" != "0" ]; then
  ARGS="$ARGS --limit-folds $LIMIT_FOLDS"
fi
if [ "$RESUME" = "0" ]; then
  ARGS="$ARGS --no-resume"
fi

run_python \
  "$ROOT_DIR/scripts/eval/run_cnn_clinical_loocv.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --k-values "$K_VALUES" \
  --seeds "$SEEDS" \
  --conditions "$CONDITIONS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  $ARGS
