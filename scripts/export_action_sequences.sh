#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CASES="${CASES:?CASES must point to an eval cases.jsonl file}"
OUTPUT="${OUTPUT:-${CASES%/*}/action_sequences.jsonl}"

python eval/action_sequences.py --cases "$CASES" --output "$OUTPUT" "$@"
