#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

OUTDIR="${OUTDIR:-$ROOT_DIR/data/simulator_qrs}"
PATIENTS_PER_SOURCE_FAMILY="${PATIENTS_PER_SOURCE_FAMILY:-20}"
SEED="${SEED:-2026}"
K_VALUES="${K_VALUES:-2,4,8,16,32}"
SEEDS="${SEEDS:-42,123,2026}"
VAL_PER_CLASS="${VAL_PER_CLASS:-4}"

. "$ROOT_DIR/scripts/run/_common.sh"

OVERWRITE="${OVERWRITE:-0}"
set --
if [ "$OVERWRITE" = "1" ]; then
  set -- --overwrite
fi

run_ecg build-synthetic \
  --outdir "$OUTDIR" \
  --patients-per-source-family "$PATIENTS_PER_SOURCE_FAMILY" \
  --seed "$SEED" \
  --k-values "$K_VALUES" \
  --seeds "$SEEDS" \
  --val-per-class "$VAL_PER_CLASS" \
  "$@"
