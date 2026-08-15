#!/usr/bin/env bash
# Smoke: CoT next-token gender probs — 2 conditions × 2 prompts (no free gen).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
RUN_ID="${RUN_ID:-qwen_cot_gender_probs_smoke}"
SCORE_POINT="${SCORE_POINT:-forced_suffix}"

python3 cot_gender_baseline_probe.py \
  --model-name "${MODEL_NAME}" \
  --run-id "${RUN_ID}" \
  --score-point "${SCORE_POINT}" \
  --conditions asthma,depression \
  --prompt-ids 1,11 \
  --n-repeats 1 \
  "$@"
