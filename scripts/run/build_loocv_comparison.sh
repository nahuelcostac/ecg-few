#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

CNN_SUMMARY="${CNN_SUMMARY:-$ROOT_DIR/outputs/work/cnn_loocv/reports/cnn_summary_by_seed.csv}"
VLM_SUMMARY="${VLM_SUMMARY:-$ROOT_DIR/outputs/work/vlm_loocv/reports/vlm_summary_by_seed.csv}"
VLM_CONDITION="${VLM_CONDITION:-estandar}"
VLM_MODEL="${VLM_MODEL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/work/comparison/reports}"

. "$ROOT_DIR/scripts/run/_common.sh"

run_python \
  "$ROOT_DIR/scripts/eval/compare_loocv_reports.py" \
  --cnn-summary "$CNN_SUMMARY" \
  --vlm-summary "$VLM_SUMMARY" \
  --vlm-condition "$VLM_CONDITION" \
  --vlm-model "$VLM_MODEL" \
  --output-dir "$OUTPUT_DIR"
