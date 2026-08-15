#!/usr/bin/env bash
# Full CoT behavioral: 5 cohorts × 20 prompts × 35 gens (no patching).
# Expect several hours on A10 — use tmux.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
RUN_ID="${RUN_ID:-qwen_cot_behavioral_n35}"

python3 generate_cot_behavioral_vignettes.py \
  --model-name "${MODEL_NAME}" \
  --run-id "${RUN_ID}" \
  --output-dir vignette_results \
  --cohorts asthma,depression,multiple_sclerosis,rheumatoid_arthritis,sarcoidosis \
  --prompt-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --n-per-cell 35 \
  --max-new-tokens 320 \
  --temperature 0.7 \
  --resume \
  "$@"
