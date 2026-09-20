"""Compatibility patches for running PackLab SFT on the upstream verl stack."""

from train.compat.distributed import bind_local_cuda_device, patch_process_group_device_binding
from train.compat.qwen35 import (
    normalize_qwen3_5_position_ids,
    patch_qwen3_5_fsdp_inputs,
    patch_qwen3_5_image_feature_dtype,
    patch_qwen3_5_model_position_ids,
    patch_qwen3_5_rotary_position_ids,
    patch_qwen3_5_text_model_position_ids,
)
from train.compat.sft_runtime import patch_sft_trainer_keep_last_batches


def apply_verl_compat_patches() -> None:
    bind_local_cuda_device()
    patch_process_group_device_binding()
    patch_qwen3_5_fsdp_inputs()
    patch_qwen3_5_model_position_ids()
    patch_qwen3_5_text_model_position_ids()
    patch_qwen3_5_image_feature_dtype()
    patch_qwen3_5_rotary_position_ids()
    patch_sft_trainer_keep_last_batches()
