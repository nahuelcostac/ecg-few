#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/simulator_qrs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/work/cnn_simulator_qrs_loocv}"
REPORT_DIR="${REPORT_DIR:-$OUTPUT_ROOT/reports}"
K_VALUES="${K_VALUES:-2,4,8,16,32}"
SEEDS="${SEEDS:-42,123,2026}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-32}"
DEVICE="${DEVICE:-auto}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
RESNET_WEIGHTS="${RESNET_WEIGHTS:-default}"
GRADCAM_COUNT="${GRADCAM_COUNT:-12}"
LIMIT_FOLDS="${LIMIT_FOLDS:-0}"
RESUME="${RESUME:-1}"

. "$ROOT_DIR/scripts/run/_common.sh"

ARGS=""
if [ "$LIMIT_FOLDS" != "0" ]; then
  ARGS="$ARGS --limit-folds $LIMIT_FOLDS"
fi
if [ "$RESUME" = "0" ]; then
  ARGS="$ARGS --no-resume"
fi

run_python \
  "$ROOT_DIR/scripts/eval/run_cnn_loocv.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --report-dir "$REPORT_DIR" \
  --k-values "$K_VALUES" \
  --seeds "$SEEDS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --image-size "$IMAGE_SIZE" \
  --resnet-weights "$RESNET_WEIGHTS" \
  --gradcam-count "$GRADCAM_COUNT" \
  $ARGS
