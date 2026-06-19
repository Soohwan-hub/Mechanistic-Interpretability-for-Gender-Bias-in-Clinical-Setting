#!/usr/bin/env bash
# Qwen Option B: paper interchange + FREE prompts 1–31 + canonical p0.
# patch-subtoken last, condition-occurrence first, layer 18, factors 0–5, n35.
#
# Usage (GPU):
#   cd activation_patching/simple_patching && source ../../.venv/bin/activate
#   ./run_qwen_optionB_interchange_free_canonical_n35.sh
#
# Resume:
#   ./run_qwen_optionB_interchange_free_canonical_n35.sh --resume
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
RUN_ID="${RUN_ID:-qwen_optionB_interchange_free_canonical_layer18_n35}"
OUTPUT_DIR="${OUTPUT_DIR:-vignette_results}"
COHORTS="${COHORTS:-asthma,depression,multiple_sclerosis,rheumatoid_arthritis,sarcoidosis}"
PROMPT_IDS="${PROMPT_IDS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"

python3 generate_scaled_vignettes.py \
  --paper-interchange-setup \
  --paper-interchange-prompt-source free \
  --include-interchange-canonical \
  --prompt-ids "${PROMPT_IDS}" \
  --cohorts "${COHORTS}" \
  --layer 18 \
  --window 0 \
  --patch-subtoken last \
  --condition-occurrence first \
  --factors 1,2,3,4,5 \
  --include-baseline \
  --outer-n 5 \
  --inner-n 7 \
  --max-new-tokens 80 \
  --temperature 0.7 \
  --target-gender Male \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  "$@"
