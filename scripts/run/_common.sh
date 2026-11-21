#!/bin/sh

UV="${UV:-uv}"
MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLBACKEND

run_ecg() {
  if command -v "$UV" >/dev/null 2>&1; then
    "$UV" run --project "$ROOT_DIR" --no-sync ecg-few "$@"
  else
    python -m ecg_few "$@"
  fi
}
