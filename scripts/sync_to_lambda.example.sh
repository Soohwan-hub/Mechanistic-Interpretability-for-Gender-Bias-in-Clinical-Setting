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

# MINIMAL_SYNC=1: only activation_patching/simple_patching/ (enough for generate_scaled_vignettes.py).
# Full repo sync is only needed if you want vendored paper code, datasets, or git history on Lambda.
MINIMAL_SYNC="${MINIMAL_SYNC:-0}"

# For full-tree sync, skip bulky dirs you do not need to *run* the vignette script (reference only).
EXCLUDE_VENDORED="${EXCLUDE_VENDORED:-0}"

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
if [[ "${EXCLUDE_VENDORED}" == "1" ]]; then
  RSYNC_EXCLUDES+=(
    --exclude='interp-healthcare-bias/'
    --exclude='ai110-module5tinker-bughound-starter/'
    --exclude='.git/'
  )
fi

REMOTE_PATCHING="${REMOTE_DIR}/activation_patching/simple_patching"

if [[ "${MINIMAL_SYNC}" == "1" ]]; then
  ssh "${LAMBDA_USER}@${LAMBDA_HOST}" "mkdir -p ${REMOTE_PATCHING}"
  rsync -avz --progress \
    "${RSYNC_EXCLUDES[@]}" \
    -e ssh \
    "${ROOT}/activation_patching/simple_patching/" \
    "${LAMBDA_USER}@${LAMBDA_HOST}:${REMOTE_PATCHING}/"
else
  # incremental sync; omit --delete if you do not want remote files removed when local deletes
  rsync -avz --progress \
    "${RSYNC_EXCLUDES[@]}" \
    -e ssh \
    "${ROOT}/" \
    "${LAMBDA_USER}@${LAMBDA_HOST}:${REMOTE_DIR}/"
fi

echo "Done. On Lambda: cd ${REMOTE_DIR} && python3 -m venv .venv && source .venv/bin/activate && pip install -r ..."
