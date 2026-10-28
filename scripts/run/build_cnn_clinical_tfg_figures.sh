#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
UV="${UV:-uv}"
CANONICAL_ROOT="${CANONICAL_ROOT:-$ROOT_DIR/outputs/final_results_tfg/loocv}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/final_results_tfg/tfg_figures/results}"
THESIS_FIGURES_DIR="${THESIS_FIGURES_DIR:-$ROOT_DIR/thesis/thesis/assets/results}"
THESIS_GENERATED_DIR="${THESIS_GENERATED_DIR:-$ROOT_DIR/thesis/thesis/generated}"

PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
MPLBACKEND="${MPLBACKEND:-Agg}"
export PYTHONPATH MPLBACKEND

if command -v "$UV" >/dev/null 2>&1; then
  "$UV" run --no-sync python \
    "$ROOT_DIR/scripts/thesis/render_clinical_cnn_figures.py" \
    --canonical-root "$CANONICAL_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --thesis-figures-dir "$THESIS_FIGURES_DIR" \
    --thesis-generated-dir "$THESIS_GENERATED_DIR"
else
  python "$ROOT_DIR/scripts/thesis/render_clinical_cnn_figures.py" \
    --canonical-root "$CANONICAL_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --thesis-figures-dir "$THESIS_FIGURES_DIR" \
    --thesis-generated-dir "$THESIS_GENERATED_DIR"
fi

printf "\n[OK] Clinical CNN figures copied to: %s\n" "$THESIS_FIGURES_DIR"
