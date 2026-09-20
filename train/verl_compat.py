"""Public compatibility entrypoint for the vendored verl stack."""

from train.compat import apply_verl_compat_patches, normalize_qwen3_5_position_ids

__all__ = ["apply_verl_compat_patches", "normalize_qwen3_5_position_ids"]
