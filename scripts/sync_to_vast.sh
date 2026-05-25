#!/usr/bin/env bash
# One-shot sync of the minimum files needed to run the baseline + biasing + pseudo-label
# jobs on a vast.ai instance.
#
# Usage:
#   export VAST_HOST=ssh4.vast.ai
#   export VAST_PORT=12345
#   export VAST_DEST=/workspace/praxis
#   export PRAXIS_DATA_ROOT=$PWD        # local source root (defaults to $PWD)
#   bash scripts/sync_to_vast.sh
#
# Requires: rsync, ssh key already added to the vast instance.

set -euo pipefail

: "${VAST_HOST:?set VAST_HOST (e.g. ssh4.vast.ai)}"
: "${VAST_PORT:?set VAST_PORT (e.g. 12345)}"
: "${VAST_DEST:=/workspace/praxis}"
: "${PRAXIS_DATA_ROOT:=$PWD}"

echo "Syncing from ${PRAXIS_DATA_ROOT} to root@${VAST_HOST}:${VAST_PORT} -> ${VAST_DEST}"
echo

cd "${PRAXIS_DATA_ROOT}"
rsync -avz --progress --relative \
  -e "ssh -p ${VAST_PORT} -o StrictHostKeyChecking=accept-new" \
  scripts/baseline_sweep_primock57.py \
  scripts/snomed_biasing_poc.py \
  scripts/pseudo_label_cv_engb_poc.py \
  scripts/wer_primock57.py \
  data/models/conformer_v2/best_primock57.pt \
  data/models/conformer_v2/praxis_tokenizer_v2.model \
  data/processed/primock57_chunked/ \
  data/raw/primock57/audio/ \
  "root@${VAST_HOST}:${VAST_DEST}/"

echo
echo "Manifests are repo-relative; resolved on the box via PRAXIS_DATA_ROOT=${VAST_DEST}. Verifying upload..."
ssh -p "${VAST_PORT}" "root@${VAST_HOST}" \
  "echo 'Sanity check:' && \
   ls ${VAST_DEST}/data/raw/primock57/audio/ | head -3 && \
   ls ${VAST_DEST}/data/models/conformer_v2/ && \
   ls ${VAST_DEST}/scripts/"

echo
echo "=== Sync complete. On vast, run: ==="
echo "  cd ${VAST_DEST}"
echo "  export PRAXIS_DATA_ROOT=${VAST_DEST}"
echo "  python3 scripts/baseline_sweep_primock57.py --model openai/whisper-small    --device cuda && \\"
echo "  python3 scripts/baseline_sweep_primock57.py --model openai/whisper-large-v3 --device cuda && \\"
echo "  python3 scripts/snomed_biasing_poc.py --device cuda --no-biasing --tag baseline_greedy && \\"
echo "  python3 scripts/snomed_biasing_poc.py --device cuda --boost 8.0  --tag biased_boost8 && \\"
echo "  python3 scripts/pseudo_label_cv_engb_poc.py --hours 20 --device cuda --save_audio"
