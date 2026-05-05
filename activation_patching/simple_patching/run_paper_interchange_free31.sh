#!/usr/bin/env bash
# Paper-style interchange replication + 31 FREE prompt variants (same single-message
# packing + token anchor + diff as interp-healthcare-bias get_interchange_accuracy.py).
#
# Requires: GPU, Python env with nnsight, torch, transformers, etc.
#
# Usage:
#   chmod +x run_paper_interchange_free31.sh
#   ./run_paper_interchange_free31.sh              # Qwen 2.5 7B (default below)
#   MODEL_NAME=allenai/OLMo-7B-0724-Instruct-hf ./run_paper_interchange_free31.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
RUN_ID="${RUN_ID:-paper_pack_free31_layer18}"
# Relative to this script's directory (we cd there before invoking Python).
OUTPUT_DIR="${OUTPUT_DIR:-vignette_results}"
COHORTS="${COHORTS:-asthma,depression,multiple_sclerosis,rheumatoid_arthritis,sarcoidosis}"
PROMPT_IDS="${PROMPT_IDS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"

python3 generate_scaled_vignettes.py \
  --paper-interchange-setup \
  --paper-interchange-prompt-source free \
  --prompt-ids "${PROMPT_IDS}" \
  --cohorts "${COHORTS}" \
  --layer 18 \
  --window 0 \
  --factors 1,2,5 \
  --outer-n 25 \
  --inner-n 20 \
  --max-new-tokens 80 \
  --temperature 0.7 \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  "$@"
