"""Distributed runtime compatibility for single-node torchrun SFT."""

from __future__ import annotations

import os
from functools import wraps


PATCH_ATTR = "_packlab_device_binding_patch"


def bind_local_cuda_device() -> None:
    """Bind this torchrun rank to its local CUDA device before collectives."""
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        return
    try:
        import torch
    except Exception:
        return
    if torch.cuda.is_available():
        torch.cuda.set_device(int(local_rank))


def patch_process_group_device_binding() -> None:
    """Ensure the fixed verl SFT stack binds CUDA before initializing NCCL."""
    try:
        import torch.distributed
        import verl.utils.distributed as distributed_utils
    except Exception:
        return
    original = distributed_utils.initialize_global_process_group_ray
    if getattr(original, PATCH_ATTR, False):
        return

    @wraps(original)
    def wrapped_initialize_global_process_group_ray(*args, **kwargs):
        bind_local_cuda_device()
        result = original(*args, **kwargs)
        bind_local_cuda_device()
        return result

    setattr(wrapped_initialize_global_process_group_ray, PATCH_ATTR, True)
    distributed_utils.initialize_global_process_group_ray = wrapped_initialize_global_process_group_ray
    _patch_default_barrier_device(torch.distributed)


def _patch_default_barrier_device(torch_distributed) -> None:
    original = torch_distributed.barrier
    barrier_attr = f"{PATCH_ATTR}_barrier"
    if getattr(original, barrier_attr, False):
        return

    @wraps(original)
    def wrapped_barrier(*args, **kwargs):
        if "device_ids" not in kwargs and _is_default_cuda_collective_group(torch_distributed):
            local_rank = os.environ.get("LOCAL_RANK")
            if local_rank is not None:
                kwargs["device_ids"] = [int(local_rank)]
        return original(*args, **kwargs)

    setattr(wrapped_barrier, barrier_attr, True)
    torch_distributed.barrier = wrapped_barrier


def _is_default_cuda_collective_group(torch_distributed) -> bool:
    if not torch_distributed.is_available() or not torch_distributed.is_initialized():
        return False
    try:
        backend = torch_distributed.get_backend()
    except Exception:
        return False
    return "nccl" in str(backend).lower()
