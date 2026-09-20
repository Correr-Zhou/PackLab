#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

CONFIG="${CONFIG:-configs/data/packlab_bench.yaml}"
python data_engine/build_dataset.py --config "$CONFIG" "$@"
