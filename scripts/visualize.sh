#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ACTION_SEQUENCES="${ACTION_SEQUENCES:?ACTION_SEQUENCES must point to action_sequences.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/visualization}"

python eval/visualize.py \
  --action-sequences "$ACTION_SEQUENCES" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
