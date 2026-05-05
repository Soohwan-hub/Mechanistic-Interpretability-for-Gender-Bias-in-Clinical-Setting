#!/usr/bin/env bash
# Copy this repo to a Lambda (or any SSH) host without using git push.
#
#   cp scripts/sync_to_lambda.example.sh scripts/sync_to_lambda.sh
#   edit LAMBDA_* below, then:
#   chmod +x scripts/sync_to_lambda.sh && ./scripts/sync_to_lambda.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --- edit these ---
LAMBDA_USER="${LAMBDA_USER:-ubuntu}"
LAMBDA_HOST="${LAMBDA_HOST:?Set LAMBDA_HOST to your instance IP or hostname}"
REMOTE_DIR="${REMOTE_DIR:-~/Mechanistic-Interpretability-for-Gender-Bias-in-Clinical-Setting}"

# Set to 0 to include local vignette_results (large); default skips them — regenerate on Lambda.
EXCLUDE_VIGNETTE_RESULTS="${EXCLUDE_VIGNETTE_RESULTS:-1}"

RSYNC_EXCLUDES=(
  --exclude='.venv/'
  --exclude='venv/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='.DS_Store'
)
if [[ "${EXCLUDE_VIGNETTE_RESULTS}" == "1" ]]; then
  RSYNC_EXCLUDES+=(--exclude='vignette_results/')
fi

# incremental sync; omit --delete if you do not want remote files removed when local deletes
rsync -avz --progress \
  "${RSYNC_EXCLUDES[@]}" \
  -e ssh \
  "${ROOT}/" \
  "${LAMBDA_USER}@${LAMBDA_HOST}:${REMOTE_DIR}/"

echo "Done. On Lambda: cd ${REMOTE_DIR} && python3 -m venv .venv && source .venv/bin/activate && pip install -r ..."
