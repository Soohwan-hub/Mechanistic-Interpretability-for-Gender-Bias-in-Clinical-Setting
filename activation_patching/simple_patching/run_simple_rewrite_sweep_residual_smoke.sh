#!/usr/bin/env bash
# Smoke test: residual-stream rewrite sweep on 2 cohorts × 2 prompts.
#
# Usage:
#   cd activation_patching/simple_patching && source ../../.venv/bin/activate
#   ./run_simple_rewrite_sweep_residual_smoke.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_ID="${RUN_ID:-qwen_simple_residual_rewrite_smoke}" \
COHORTS="${COHORTS:-asthma,multiple_sclerosis}" \
PROMPT_IDS="${PROMPT_IDS:-1,5}" \
exec ./run_simple_rewrite_sweep_residual.sh \
  --layer-step 2 \
  --max-tokens 80 \
  "$@"
