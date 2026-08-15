#!/usr/bin/env bash
# Full: CoT next-token gender probs — 5 conditions × 20 prompts (Fig-1 style).
# No free vignettes; forced Gender: then P(Male)/P(Female).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
RUN_ID="${RUN_ID:-qwen_cot20_gender_probs}"
SCORE_POINT="${SCORE_POINT:-forced_suffix}"

python3 cot_gender_baseline_probe.py \
  --model-name "${MODEL_NAME}" \
  --run-id "${RUN_ID}" \
  --score-point "${SCORE_POINT}" \
  --conditions "asthma,depression,multiple sclerosis,rheumatoid arthritis,sarcoidosis" \
  --prompt-ids 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20 \
  --n-repeats 1 \
  "$@"
