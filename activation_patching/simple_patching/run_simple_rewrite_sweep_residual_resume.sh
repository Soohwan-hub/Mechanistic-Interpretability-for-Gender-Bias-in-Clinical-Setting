#!/usr/bin/env bash
# Resume qwen_simple31_residual_rewrite (skips completed artifacts/*.pkl).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
exec ./run_simple_rewrite_sweep_residual.sh --resume "$@"
