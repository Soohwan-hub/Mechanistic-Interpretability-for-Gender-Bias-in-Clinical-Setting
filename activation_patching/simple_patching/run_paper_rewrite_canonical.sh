#!/usr/bin/env bash
# Paper-faithful rewrite-score sweep (get_patching_scores.py):
#   - canonical gender template (one wording per model)
#   - single chat turn: "You are a helpful clinical assistant." + task body
#   - clean prompt: "The patient is Male." with model-specific role
#   - 5 cohorts × 1 prompt = 5 pkls + heatmaps
#
# Usage (Lambda / GPU):
#   cd activation_patching/simple_patching && source ../../.venv/bin/activate
#   ./run_paper_rewrite_canonical.sh
#
# Resume:
#   ./run_paper_rewrite_canonical_resume.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_ID="${MODEL_ID:-2}"
RUN_ID="${RUN_ID:-olmo_paper_rewrite_canonical}"
OUTPUT_DIR="${OUTPUT_DIR:-patching_results}"
COHORTS="${COHORTS:-asthma,depression,multiple_sclerosis,rheumatoid_arthritis,sarcoidosis}"
# OLMo paper batches clean activations in chunks of 5 layers (memory tuning).
LAYER_STEP="${LAYER_STEP:-5}"

python3 paper_rewrite_canonical.py \
  --model-id "${MODEL_ID}" \
  --run-id "${RUN_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --cohorts "${COHORTS}" \
  --score-keys rewrite_scores \
  --layer-step "${LAYER_STEP}" \
  --save-heatmaps \
  --save-layer-plots \
  --plot-format pdf \
  --exclude-plot-layers 0 \
  "$@"
