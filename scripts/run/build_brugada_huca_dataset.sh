#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"

RAW_ROOT="${RAW_ROOT:-$ROOT_DIR/data/raw/brugada-huca/1.0.0}"
OUTDIR="${OUTDIR:-$ROOT_DIR/data/brugada_huca}"
K_VALUES="${K_VALUES:-2,4,8,16,32}"
SEEDS="${SEEDS:-42,123,2026}"
VAL_PER_CLASS="${VAL_PER_CLASS:-4}"
DPI="${DPI:-130}"
PRE_R_MS="${PRE_R_MS:-300}"
POST_R_MS="${POST_R_MS:-600}"

. "$ROOT_DIR/scripts/run/_common.sh"

OVERWRITE="${OVERWRITE:-0}"
set --
if [ "$OVERWRITE" = "1" ]; then
  set -- --overwrite
fi

run_ecg build-huca \
  --raw-root "$RAW_ROOT" \
  --outdir "$OUTDIR" \
  --k-values "$K_VALUES" \
  --seeds "$SEEDS" \
  --val-per-class "$VAL_PER_CLASS" \
  --dpi "$DPI" \
  --pre-r-ms "$PRE_R_MS" \
  --post-r-ms "$POST_R_MS" \
  "$@"
