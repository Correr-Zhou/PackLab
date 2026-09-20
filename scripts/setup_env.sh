#!/usr/bin/env bash
set -euo pipefail

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

setup_cuda_build_env() {
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
    export CUDA_HOME="${CUDA_HOME:-$CONDA_PREFIX}"
    export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
    export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
    export PATH="$CUDA_HOME/bin:$PATH"
  fi
  export CC="${CC:-/usr/bin/gcc}"
  export CXX="${CXX:-/usr/bin/g++}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++}"
}

setup_model_cache_env() {
  local cache_root="${PACKLAB_CACHE_ROOT:-/tmp/packlab_cache/${USER:-unknown}}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$cache_root/xdg}"
  export HF_HOME="${HF_HOME:-$cache_root/huggingface}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
  export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export TORCH_HOME="${TORCH_HOME:-$cache_root/torch}"
  export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$cache_root/nv_cuda}"
  export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$cache_root/triton}"
  mkdir -p "$XDG_CACHE_HOME" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE" \
    "$HF_XET_CACHE" "$TORCH_HOME" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"
}

setup_local_proxy_bypass() {
  local local_hosts="localhost,127.0.0.1,::1,0.0.0.0"
  local host_ip
  host_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
  if [[ -n "$host_ip" ]]; then
    local_hosts="${local_hosts},${host_ip}"
  fi
  export NO_PROXY="${local_hosts}${NO_PROXY:+,$NO_PROXY}"
  export no_proxy="$NO_PROXY"
}

setup_single_node_nccl() {
  if [[ "${NNODES:-1}" != "1" || "${PACKLAB_DISABLE_SINGLE_NODE_NCCL_DEFAULTS:-0}" == "1" ]]; then
    return
  fi
  export NCCL_NET_PLUGIN="${PACKLAB_NCCL_NET_PLUGIN:-none}"
  export NCCL_IB_DISABLE="${PACKLAB_NCCL_IB_DISABLE:-1}"
  export NCCL_SOCKET_FAMILY="${PACKLAB_NCCL_SOCKET_FAMILY:-AF_INET}"
  export NCCL_SOCKET_IFNAME="${PACKLAB_NCCL_SOCKET_IFNAME:-lo}"
  export GLOO_SOCKET_IFNAME="${PACKLAB_GLOO_SOCKET_IFNAME:-lo}"
  unset NCCL_FASTRAK_CTRL_DEV
  unset NCCL_FASTRAK_DATA_TRANSFER_TIMEOUT_MS
  unset NCCL_FASTRAK_DUMP_COMM_STATS
  unset NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL
  unset NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING
  unset NCCL_FASTRAK_IFNAME
  unset NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY
  unset NCCL_FASTRAK_NUM_FLOWS
  unset NCCL_FASTRAK_USE_LLCM
  unset NCCL_FASTRAK_USE_SNAP
  unset NCCL_TUNER_CONFIG_PATH
  unset NCCL_TUNER_PLUGIN
}

setup_vllm_server_env() {
  export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
  export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
  export VLLM_DISABLE_PYNCCL="${VLLM_DISABLE_PYNCCL:-1}"
  export NCCL_NET_PLUGIN="${PACKLAB_VLLM_NCCL_NET_PLUGIN:-none}"
  export NCCL_IB_DISABLE="${PACKLAB_VLLM_NCCL_IB_DISABLE:-1}"
  export NCCL_SOCKET_FAMILY="${PACKLAB_VLLM_NCCL_SOCKET_FAMILY:-AF_INET}"
  export NCCL_SOCKET_IFNAME="${PACKLAB_VLLM_NCCL_SOCKET_IFNAME:-lo}"
  export GLOO_SOCKET_IFNAME="${PACKLAB_VLLM_GLOO_SOCKET_IFNAME:-lo}"
  unset NCCL_FASTRAK_CTRL_DEV
  unset NCCL_FASTRAK_DATA_TRANSFER_TIMEOUT_MS
  unset NCCL_FASTRAK_DUMP_COMM_STATS
  unset NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL
  unset NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING
  unset NCCL_FASTRAK_IFNAME
  unset NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY
  unset NCCL_FASTRAK_NUM_FLOWS
  unset NCCL_FASTRAK_USE_LLCM
  unset NCCL_FASTRAK_USE_SNAP
  unset NCCL_TUNER_CONFIG_PATH
  unset NCCL_TUNER_PLUGIN
}
