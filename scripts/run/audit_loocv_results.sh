#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

RAW_ROOT="${RAW_ROOT:-$ROOT_DIR/data/raw/brugada-huca/1.0.0}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT_DIR/data/brugada_huca}"
CNN_REPORT_DIR="${CNN_REPORT_DIR:-$ROOT_DIR/outputs/work/cnn_loocv/reports}"
VLM_REPORT_DIR="${VLM_REPORT_DIR:-$ROOT_DIR/outputs/work/vlm_loocv/reports}"
COMPARISON_DIR="${COMPARISON_DIR:-$ROOT_DIR/outputs/work/comparison/reports}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/work/audit/reports}"
K_VALUES="${K_VALUES:-2,4,8,16,32}"
SEEDS="${SEEDS:-42,123,2026}"

. "$ROOT_DIR/scripts/run/_common.sh"

run_ecg audit \
  --raw-root "$RAW_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --cnn-report-dir "$CNN_REPORT_DIR" \
  --vlm-report-dir "$VLM_REPORT_DIR" \
  --comparison-dir "$COMPARISON_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --k-values "$K_VALUES" \
  --seeds "$SEEDS"
