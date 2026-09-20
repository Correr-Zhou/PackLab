#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_env.sh
setup_cuda_build_env
setup_model_cache_env
setup_local_proxy_bypass
export NNODES="${NNODES:-1}"
setup_single_node_nccl
setup_vllm_server_env

CONFIG="${CONFIG:-configs/eval/packlab_bench.yaml}"
MODEL_PATH="${MODEL_PATH:-checkpoints/PackLab-VLM-9B}"
API_KEY="${API_KEY:-${OPENAI_API_KEY:-${PACKLAB_OPENAI_API_KEY:-EMPTY}}}"
RUN_NAME="${RUN_NAME:-packlab_bench_$(date +%Y%m%d_%H%M%S)}"
HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
KEEP_SERVER="${KEEP_SERVER:-0}"
SERVER_STARTUP_TIMEOUT="${SERVER_STARTUP_TIMEOUT:-1200}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---runner generate --max-model-len ${MAX_MODEL_LEN} --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} --trust-remote-code --disable-custom-all-reduce --gdn-prefill-backend triton}"
LOG_DIR="${LOG_DIR:-outputs/eval/${RUN_NAME}}"
SERVER_LOG="${LOG_DIR}/server.log"
EVAL_LOG="${LOG_DIR}/eval.log"

if ! python - <<'PY' >/dev/null 2>&1
import vllm  # noqa: F401
PY
then
  echo "The active Python environment cannot import vLLM." >&2
  echo "Install the evaluation requirements and activate that environment before running this script." >&2
  exit 1
fi

read -r -a vllm_args <<< "$VLLM_EXTRA_ARGS"
if [[ "$VLLM_EXTRA_ARGS" != *"--default-chat-template-kwargs"* ]]; then
  vllm_args+=(--default-chat-template-kwargs '{"enable_thinking": false}')
fi

mkdir -p "$LOG_DIR"

cleanup() {
  if [[ "${KEEP_SERVER}" != "1" && -n "${SERVER_PID:-}" ]]; then
    local server_pgid
    server_pgid="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ' || true)"
    if [[ -n "$server_pgid" ]]; then
      kill -- "-${server_pgid}" >/dev/null 2>&1 || true
      sleep 2
      kill -KILL -- "-${server_pgid}" >/dev/null 2>&1 || true
    else
      kill "$SERVER_PID" >/dev/null 2>&1 || true
    fi
  fi
}
trap cleanup EXIT

echo "Starting vLLM server for ${MODEL_PATH} on ${HOST}:${PORT}"
setsid python -u -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --host "$HOST" \
  --port "$PORT" \
  "${vllm_args[@]}" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "Waiting for vLLM health check..."
for _ in $(seq 1 "$SERVER_STARTUP_TIMEOUT"); do
  if curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

if ! curl -fsS "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
  echo "vLLM server did not become ready. See ${SERVER_LOG}" >&2
  exit 1
fi

python eval/vllm_runner.py \
  --config "$CONFIG" \
  --base-url "http://${HOST}:${PORT}/v1" \
  --model "$MODEL_PATH" \
  --api-key "$API_KEY" \
  --run-name "$RUN_NAME" \
  prompt.allow_thinking=false \
  "$@" \
  > "$EVAL_LOG" 2>&1

cat "$EVAL_LOG"
