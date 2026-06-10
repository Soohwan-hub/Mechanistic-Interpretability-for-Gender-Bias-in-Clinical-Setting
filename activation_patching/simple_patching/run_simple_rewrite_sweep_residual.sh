#!/usr/bin/env bash
# Full simple-prompt rewrite sweep on the residual stream (Qwen2.5-7B-Instruct).
# Mirrors prior MLP runs (e.g. female5_patch_male / olmo31_rewrite_only) but with
# --patch-target residual instead of default mlp down_proj.
#
# Work units: 5 cohorts × 31 prompts = 155 cells.
#
# Usage (Lambda / GPU):
#   cd activation_patching/simple_patching && source ../../.venv/bin/activate
#   ./run_simple_rewrite_sweep_residual.sh
#
# Resume:
#   ./run_simple_rewrite_sweep_residual_resume.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${RUN_ID:-qwen_simple31_residual_rewrite}"
OUTPUT_DIR="${OUTPUT_DIR:-patching_results}"
COHORTS="${COHORTS:-asthma,depression,multiple_sclerosis,rheumatoid_arthritis,sarcoidosis}"
PROMPT_IDS="${PROMPT_IDS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"

python3 simple_patching_without_BHCs.py \
  --run-id "${RUN_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --patch-target residual \
  --cohorts "${COHORTS}" \
  --prompt-ids "${PROMPT_IDS}" \
  --score-keys rewrite_scores \
  --max-tokens 0 \
  --layer-start 0 \
  --layer-end 9999 \
  --layer-step 1 \
  "$@"
