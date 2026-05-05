#!/usr/bin/env bash
# Smoke test: same layout as the full ~8.4k run but only 6 rows per (cohort, prompt):
#   outer_n * inner_n * |factors| = 2 * 1 * 3 = 6  →  155 * 6 = 930 generations total.
#
# Usage (Lambda):
#   cd activation_patching/simple_patching && source ../../.venv/bin/activate
#   ./run_paper_interchange_free31_smoke.sh
#   MODEL_NAME=Qwen/Qwen2.5-7B-Instruct RUN_ID=qwen_smoke ./run_paper_interchange_free31_smoke.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME="${MODEL_NAME:-allenai/OLMo-7B-0724-Instruct-hf}"
RUN_ID="${RUN_ID:-olmo_free31_smoke_6per_cell}"
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
  --outer-n 2 \
  --inner-n 1 \
  --max-new-tokens 80 \
  --temperature 0.7 \
  --target-gender Male \
  --model-name "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" \
  "$@"
