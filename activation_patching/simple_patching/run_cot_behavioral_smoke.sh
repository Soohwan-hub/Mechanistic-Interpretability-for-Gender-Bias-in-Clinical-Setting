#!/usr/bin/env bash
# Smoke: CoT free vignettes (no patching) — 2 cohorts × 2 prompts × 3 gens.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
RUN_ID="${RUN_ID:-qwen_cot_behavioral_smoke}"

python3 generate_cot_behavioral_vignettes.py \
  --model-name "${MODEL_NAME}" \
  --run-id "${RUN_ID}" \
  --output-dir vignette_results \
  --cohorts asthma,depression \
  --prompt-ids 1,11 \
  --n-per-cell 3 \
  --max-new-tokens 256 \
  --temperature 0.7 \
  "$@"
