#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_env.sh
setup_model_cache_env
setup_local_proxy_bypass

CONFIG="${CONFIG:-configs/eval/packlab_bench.yaml}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
MODEL_PATH="${MODEL_PATH:-checkpoints/PackLab-VLM-9B}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-${PACKLAB_OPENAI_API_KEY:-EMPTY}}}"
RUN_NAME="${RUN_NAME:-packlab_bench_openai_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-outputs/eval/${RUN_NAME}}"
EVAL_LOG="${LOG_DIR}/eval.log"

mkdir -p "$LOG_DIR"

python eval/vllm_runner.py \
  --config "$CONFIG" \
  --base-url "$BASE_URL" \
  --model "$MODEL_PATH" \
  --api-key "$API_KEY" \
  --run-name "$RUN_NAME" \
  prompt.allow_thinking=false \
  "$@" \
  > "$EVAL_LOG" 2>&1

cat "$EVAL_LOG"
