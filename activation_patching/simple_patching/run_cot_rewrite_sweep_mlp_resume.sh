#!/usr/bin/env bash
# Resume CoT MLP rewrite sweep (same RUN_ID as run_cot_rewrite_sweep_mlp.sh).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec ./run_cot_rewrite_sweep_mlp.sh --resume "$@"
