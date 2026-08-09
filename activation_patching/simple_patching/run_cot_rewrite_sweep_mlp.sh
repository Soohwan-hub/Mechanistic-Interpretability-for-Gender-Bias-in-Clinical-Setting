#!/usr/bin/env bash
# CoT rewrite sweep on MLP down_proj (Qwen2.5-7B-Instruct).
# Entry point: cot_patching_without_BHCs.py (not simple_patching_without_BHCs.py).
#
# Uses <thinking>-tag CoT Types A and C (10 variants each, 20 prompts, no BHC).
# Work units: 5 cohorts × 20 prompts = 100 cells.
#
# Prompt id map:
#   1–10  = Type A (demographics-first CoT)
#   11–20 = Type C (swap-check CoT)
#
# Scoring: appends "Gender:" after chat template (same rewrite-score setup as female5_patch_male).
# For frozen corrupt prompt (no <thinking> in user message):
#   CORRUPT_MODE=frozen ./run_cot_rewrite_sweep_mlp.sh
#
# Usage (Lambda / GPU):
#   cd activation_patching/simple_patching && source ../../.venv/bin/activate
#   ./run_cot_rewrite_sweep_mlp.sh
#
# Resume:
#   ./run_cot_rewrite_sweep_mlp_resume.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${RUN_ID:-qwen_cot20_mlp_rewrite}"
OUTPUT_DIR="${OUTPUT_DIR:-patching_results}"
COHORTS="${COHORTS:-asthma,depression,multiple_sclerosis,rheumatoid_arthritis,sarcoidosis}"
PROMPT_IDS="${PROMPT_IDS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20}"
CORRUPT_MODE="${CORRUPT_MODE:-full}"

python3 cot_patching_without_BHCs.py \
  --run-id "${RUN_ID}" \
  --output-dir "${OUTPUT_DIR}" \
  --corrupt-mode "${CORRUPT_MODE}" \
  --patch-target mlp \
  --cohorts "${COHORTS}" \
  --prompt-ids "${PROMPT_IDS}" \
  --score-keys rewrite_scores \
  --max-tokens 0 \
  --layer-start 0 \
  --layer-end 9999 \
  --layer-step 1 \
  "$@"
