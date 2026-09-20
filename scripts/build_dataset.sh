#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/data/packdata_20k.yaml}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_VAL="${RUN_VAL:-1}"

train_args=()
[[ -n "${START_SHARD:-}" ]] && train_args+=(--start-shard "$START_SHARD")
[[ -n "${END_SHARD:-}" ]] && train_args+=(--end-shard "$END_SHARD")
[[ -n "${NUM_SHARDS:-}" ]] && train_args+=(--num-shards "$NUM_SHARDS")
[[ "${SKIP_COMPLETED:-0}" == "1" ]] && train_args+=(--skip-completed)

if [[ "$RUN_TRAIN" == "1" ]]; then
  python data_engine/build_dataset.py --config "$CONFIG" "${train_args[@]}" "$@"
fi

if [[ "$RUN_VAL" == "1" ]]; then
  python data_engine/build_dataset.py --config "$CONFIG" mode=sft_val split=val "$@"
fi
