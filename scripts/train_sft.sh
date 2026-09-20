#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source scripts/setup_env.sh
setup_cuda_build_env
setup_model_cache_env
setup_local_proxy_bypass

if [[ -n "${ARNOLD_WORKER_NUM:-}" ]]; then
  export NNODES="${NNODES:-$ARNOLD_WORKER_NUM}"
else
  export NNODES="${NNODES:-1}"
fi

if [[ -n "${ARNOLD_ID:-}" ]]; then
  export NODE_RANK="${NODE_RANK:-$ARNOLD_ID}"
else
  export NODE_RANK="${NODE_RANK:-0}"
fi

if [[ -n "${ARNOLD_WORKER_GPU:-}" ]]; then
  export NPROC_PER_NODE="${NPROC_PER_NODE:-$ARNOLD_WORKER_GPU}"
else
  export NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_GPUS:-8}}"
fi

if [[ "$NNODES" == "1" ]]; then
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
else
  export MASTER_ADDR="${MASTER_ADDR:?MASTER_ADDR is required for multi-node training}"
fi
export MASTER_PORT="${MASTER_PORT:-29500}"
setup_single_node_nccl

CONFIG="${CONFIG:-configs/train/packing_sft.yaml}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/checkpoints/packlab_vlm_9b_sft}"
WANDB_PROJECT="${WANDB_PROJECT:-packlab_sft}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-packlab_vlm_9b_sft_$(date +%Y%m%d_%H%M%S)}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[console]}"

overrides=(
  "model.path=${MODEL_PATH}"
  "trainer.project_name=${WANDB_PROJECT}"
  "trainer.experiment_name=${WANDB_RUN_NAME}"
  "trainer.logger=${TRAINER_LOGGER}"
)

[[ -n "${SFT_LR:-}" ]] && overrides+=("optim.lr=${SFT_LR}")
[[ -n "${SFT_TRAIN_BATCH_SIZE:-}" ]] && overrides+=("data.train_batch_size=${SFT_TRAIN_BATCH_SIZE}")
[[ -n "${SFT_MICRO_BATCH_SIZE_PER_GPU:-}" ]] && overrides+=("data.micro_batch_size_per_gpu=${SFT_MICRO_BATCH_SIZE_PER_GPU}")
[[ -n "${SFT_EPOCHS:-}" ]] && overrides+=("trainer.total_epochs=${SFT_EPOCHS}")
[[ -n "${SFT_TOTAL_TRAINING_STEPS:-}" ]] && overrides+=("trainer.total_training_steps=${SFT_TOTAL_TRAINING_STEPS}")
[[ -n "${SFT_SAVE_FREQ:-}" ]] && overrides+=("trainer.save_freq=${SFT_SAVE_FREQ}")
[[ -n "${SFT_TEST_FREQ:-}" ]] && overrides+=("trainer.test_freq=${SFT_TEST_FREQ}")

mapfile -t verl_cmd < <(python train/launch_verl.py --config "$CONFIG" --print-cmd "${overrides[@]}" "$@")
if [[ "${#verl_cmd[@]}" -ne 1 ]]; then
  echo "Expected one trainer command, got ${#verl_cmd[@]}" >&2
  exit 1
fi
read -r -a cmd_parts <<< "${verl_cmd[0]}"
if [[ "${cmd_parts[0]}" != "python" || "${cmd_parts[1]}" != "-m" ]]; then
  echo "Unexpected trainer command: ${verl_cmd[0]}" >&2
  exit 1
fi

echo "Resolved trainer command:"
printf '  %q' torchrun "--nnodes=${NNODES}" "--node-rank=${NODE_RANK}" "--nproc-per-node=${NPROC_PER_NODE}" "--master-addr=${MASTER_ADDR}" "--master-port=${MASTER_PORT}" -m "${cmd_parts[@]:2}"
printf '\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

mkdir -p "$(dirname "$OUTPUT_DIR")" outputs/logs
torchrun \
  "--nnodes=${NNODES}" \
  "--node-rank=${NODE_RANK}" \
  "--nproc-per-node=${NPROC_PER_NODE}" \
  "--master-addr=${MASTER_ADDR}" \
  "--master-port=${MASTER_PORT}" \
  -m "${cmd_parts[@]:2}"
