"""Qwen3.5 VLM compatibility patches for the current verl FSDP data path.

These patches are intentionally scoped to Qwen3.5 boundaries:

- verl FSDP input preparation can hand batch-first or nested MRoPE ids to the
  Hugging Face model, while native Qwen3.5 expects channel-first ids at several
  model boundaries.
- root FSDP wrapping can hide visual tower parameters from the dtype property
  path used by Qwen3.5 image feature extraction.
- non-Qwen3.5 models must keep their original inputs and model methods.
"""

from __future__ import annotations

from functools import wraps
import os

import torch


_TRACE_KEYS_PRINTED: set[str] = set()
TRACE_QWEN35_SHAPES = "PACKLAB_TRACE_QWEN35_SHAPES"
QWEN35_PATCH_ATTR = "_packlab_qwen35_patched"


def normalize_qwen3_5_position_ids(model_inputs: dict) -> dict:
    """Normalize Qwen3.5 MRoPE position ids to Hugging Face's channel-first layout."""
    position_ids = model_inputs.get("position_ids")
    if (
        isinstance(position_ids, torch.Tensor)
        and position_ids.ndim == 3
        and position_ids.shape[0] != 4
        and position_ids.shape[1] == 4
    ):
        model_inputs = dict(model_inputs)
        model_inputs["position_ids"] = position_ids.transpose(0, 1).contiguous()
    elif isinstance(position_ids, torch.Tensor) and position_ids.ndim == 2 and position_ids.shape[0] == 4:
        model_inputs = dict(model_inputs)
        model_inputs["position_ids"] = position_ids.unsqueeze(1)
    return model_inputs


def _trace_once(key: str, message: str) -> None:
    if os.environ.get(TRACE_QWEN35_SHAPES) != "1" or key in _TRACE_KEYS_PRINTED:
        return
    _TRACE_KEYS_PRINTED.add(key)
    print(f"[packlab:qwen3_5] {message}", flush=True)


def _recover_qwen3_5_position_ids(model_inputs: dict, micro_batch) -> dict:
    input_ids = model_inputs.get("input_ids")
    position_ids = model_inputs.get("position_ids")
    if not (
        isinstance(input_ids, torch.Tensor)
        and input_ids.ndim == 2
        and isinstance(position_ids, torch.Tensor)
        and position_ids.shape[-1] != input_ids.shape[-1]
    ):
        return model_inputs

    recovered = None
    source_position_ids = micro_batch.get("position_ids", None)
    if isinstance(source_position_ids, torch.Tensor) and source_position_ids.is_nested and source_position_ids.dim() == 3:
        values = source_position_ids.values()
        if values.ndim == 2 and values.shape == (4, input_ids.shape[1]):
            recovered = values.unsqueeze(1)
        elif values.ndim == 2 and values.shape == (input_ids.shape[1], 4):
            recovered = values.transpose(0, 1).unsqueeze(1)

    if recovered is None:
        recovered = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(input_ids.shape[0], -1)

    model_inputs = dict(model_inputs)
    model_inputs["position_ids"] = recovered.to(device=position_ids.device)
    _trace_once(
        "fsdp_position_ids_recover",
        f"Recovered malformed FSDP position_ids {tuple(position_ids.shape)}"
        f" -> {tuple(model_inputs['position_ids'].shape)}",
    )
    return model_inputs


def patch_qwen3_5_fsdp_inputs() -> None:
    """Patch verl FSDP input preparation only for native HF Qwen3.5 VLM models."""
    try:
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    except Exception:
        return

    if getattr(FSDPEngineWithLMHead.prepare_model_inputs, QWEN35_PATCH_ATTR, False):
        return

    original = FSDPEngineWithLMHead.prepare_model_inputs

    @wraps(original)
    def wrapped(self, micro_batch):
        model_inputs, output_args = original(self, micro_batch)
        hf_config = getattr(getattr(self, "model_config", None), "hf_config", None)
        if getattr(hf_config, "model_type", None) == "qwen3_5":
            model_inputs = normalize_qwen3_5_position_ids(model_inputs)
            model_inputs = _recover_qwen3_5_position_ids(model_inputs, micro_batch)
            _trace_once(
                "fsdp_model_inputs",
                "FSDP model_inputs "
                + ", ".join(
                    f"{key}={tuple(value.shape)}"
                    for key, value in model_inputs.items()
                    if isinstance(value, torch.Tensor)
                ),
            )
        return model_inputs, output_args

    setattr(wrapped, QWEN35_PATCH_ATTR, True)
    FSDPEngineWithLMHead.prepare_model_inputs = wrapped


def patch_qwen3_5_text_model_position_ids() -> None:
    """Accept batch-first Qwen3.5 MRoPE ids before HF text model RoPE setup."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    except Exception:
        return

    if getattr(Qwen3_5TextModel.forward, QWEN35_PATCH_ATTR, False):
        return

    original = Qwen3_5TextModel.forward

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        args = list(args)
        if len(args) > 2 and isinstance(args[2], torch.Tensor):
            before = args[2]
            args[2] = normalize_qwen3_5_position_ids({"position_ids": args[2]})["position_ids"]
            _trace_once(
                "text_position_ids",
                f"TextModel position_ids {tuple(before.shape)} -> {tuple(args[2].shape)}",
            )
        if "position_ids" in kwargs:
            before = kwargs["position_ids"]
            kwargs["position_ids"] = normalize_qwen3_5_position_ids({"position_ids": kwargs["position_ids"]})[
                "position_ids"
            ]
            after = kwargs["position_ids"]
            _trace_once(
                "text_position_ids",
                f"TextModel position_ids {tuple(before.shape) if isinstance(before, torch.Tensor) else None}"
                f" -> {tuple(after.shape) if isinstance(after, torch.Tensor) else None}",
            )
        return original(self, *args, **kwargs)

    setattr(wrapped, QWEN35_PATCH_ATTR, True)
    Qwen3_5TextModel.forward = wrapped


def _first_floating_dtype(module) -> torch.dtype:
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.get_default_dtype()


def patch_qwen3_5_image_feature_dtype() -> None:
    """Keep Qwen3.5 image preprocessing working when root FSDP flattens visual params."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    except Exception:
        return

    if getattr(Qwen3_5Model.get_image_features, QWEN35_PATCH_ATTR, False):
        return

    original = Qwen3_5Model.get_image_features

    @wraps(original)
    def wrapped(self, pixel_values, image_grid_thw=None, **kwargs):
        try:
            self.visual.dtype
        except StopIteration:
            visual_dtype = _first_floating_dtype(self)
            pixel_values = pixel_values.type(visual_dtype)
            try:
                vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True, **kwargs)
            except KeyError as exc:
                if exc.args != ("return_dict",):
                    raise
                vision_output = self.visual(pixel_values, grid_thw=image_grid_thw, **kwargs)
            image_embeds = vision_output.pooler_output
            split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
            vision_output.pooler_output = torch.split(image_embeds, split_sizes)
            return vision_output
        return original(self, pixel_values, image_grid_thw, **kwargs)

    setattr(wrapped, QWEN35_PATCH_ATTR, True)
    Qwen3_5Model.get_image_features = wrapped


def patch_qwen3_5_model_position_ids() -> None:
    """Let HF Qwen3.5 recompute malformed precomputed MRoPE ids."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    except Exception:
        return

    if getattr(Qwen3_5Model.forward, QWEN35_PATCH_ATTR, False):
        return

    original = Qwen3_5Model.forward

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        args = list(args)
        position_ids = kwargs.get("position_ids")
        if position_ids is None and len(args) > 2:
            position_ids = args[2]
        input_ids = kwargs.get("input_ids", args[0] if len(args) > 0 else None)
        inputs_embeds = kwargs.get("inputs_embeds", args[4] if len(args) > 4 else None)
        seq_len = None
        if isinstance(inputs_embeds, torch.Tensor):
            seq_len = inputs_embeds.shape[1]
        elif isinstance(input_ids, torch.Tensor):
            seq_len = input_ids.shape[1]
        if isinstance(position_ids, torch.Tensor) and seq_len is not None and position_ids.shape[-1] != seq_len:
            _trace_once(
                "model_position_ids_drop",
                f"Qwen3_5Model dropping malformed position_ids={tuple(position_ids.shape)} for seq_len={seq_len}",
            )
            if "position_ids" in kwargs:
                kwargs["position_ids"] = None
            elif len(args) > 2:
                args[2] = None
        return original(self, *args, **kwargs)

    setattr(wrapped, QWEN35_PATCH_ATTR, True)
    Qwen3_5Model.forward = wrapped


def patch_qwen3_5_rotary_position_ids() -> None:
    """Normalize Qwen3.5 rotary position ids at the final RoPE boundary."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    except Exception:
        return

    if getattr(Qwen3_5TextRotaryEmbedding.forward, QWEN35_PATCH_ATTR, False):
        return

    original = Qwen3_5TextRotaryEmbedding.forward

    @wraps(original)
    def wrapped(self, x, position_ids):
        before = position_ids
        if isinstance(position_ids, torch.Tensor):
            if position_ids.ndim == 2 and position_ids.shape[0] == 4:
                position_ids = position_ids[1:].unsqueeze(1)
            elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
                position_ids = position_ids[1:]
            elif position_ids.ndim == 3 and position_ids.shape[1] == 4:
                position_ids = position_ids.transpose(0, 1)[1:].contiguous()
        _trace_once(
            "rotary_position_ids",
            f"Rotary position_ids {tuple(before.shape) if isinstance(before, torch.Tensor) else None}"
            f" -> {tuple(position_ids.shape) if isinstance(position_ids, torch.Tensor) else None}; x={tuple(x.shape)}",
        )
        return original(self, x, position_ids)

    setattr(wrapped, QWEN35_PATCH_ATTR, True)
    Qwen3_5TextRotaryEmbedding.forward = wrapped
