#!/usr/bin/env bash
# Smoke test: CoT MLP rewrite sweep on 2 cohorts × 2 prompts (A_var1 + C_var1).
#
# Usage:
#   cd activation_patching/simple_patching && source ../../.venv/bin/activate
#   ./run_cot_rewrite_sweep_mlp_smoke.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${RUN_ID:-qwen_cot20_mlp_rewrite_smoke}" \
COHORTS="${COHORTS:-multiple_sclerosis,depression}" \
PROMPT_IDS="${PROMPT_IDS:-1,11}" \
exec ./run_cot_rewrite_sweep_mlp.sh \
  --layer-step 2 \
  --max-tokens 80 \
  "$@"
