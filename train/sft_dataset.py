"""Optional SFT dataset path adapter for full-sequence data."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import pandas as pd
import numpy as np


def _resolve_image_item(image_item, root: Path):
    if isinstance(image_item, dict):
        copied = dict(image_item)
        image_path = copied.get("image")
        if image_path is not None:
            path = Path(image_path)
            copied["image"] = str(path if path.is_absolute() else root / path)
        return copied
    path = Path(image_item)
    return str(path if path.is_absolute() else root / path)


def resolve_relative_images(row: dict, data_root: str | Path) -> dict:
    """Return a copy of a row with relative image paths resolved against data_root."""
    root = Path(data_root)
    copied = dict(row)
    copied["images"] = [_resolve_image_item(image_item, root) for image_item in row.get("images", [])]
    return copied


def resolve_enable_thinking_values(dataframe: pd.DataFrame, key: str) -> list[bool]:
    """Packing SFT defaults to Qwen thinking-off when rows omit the column."""
    if key in dataframe.columns:
        return dataframe[key].tolist()
    return [False] * len(dataframe)


try:
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
except Exception:
    MultiTurnSFTDataset = None


if MultiTurnSFTDataset is None:

    class PackingSequenceSFTDataset:  # pragma: no cover - exercised only when optional verl dataset is missing.
        def __init__(self, *args, **kwargs):
            raise ImportError("verl.utils.dataset.multiturn_sft_dataset is required for PackingSequenceSFTDataset")


else:
    from verl.utils.chat_template import extract_system_prompt_and_generation
    from verl.utils.py_functional import convert_nested_value_to_list_recursive

    def _as_1d_tensor(token_ids) -> torch.Tensor:
        if isinstance(token_ids, dict) and "input_ids" in token_ids:
            token_ids = token_ids["input_ids"]
        if isinstance(token_ids, torch.Tensor):
            return token_ids.flatten().to(dtype=torch.long)
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return torch.tensor(token_ids, dtype=torch.long)

    class PackingSequenceSFTDataset(MultiTurnSFTDataset):
        """Thin path adapter for full-sequence packing SFT parquet rows."""

        def _data_root_for_file(self, parquet_file: str | Path) -> Path:
            configured = self.config.get("data_root", None) if hasattr(self, "config") else None
            if configured:
                return Path(configured)
            path = Path(parquet_file)
            if "train_data_sft" in path.parts:
                return Path(*path.parts[: path.parts.index("train_data_sft") + 1])
            if len(path.parents) >= 2 and path.parent.name.startswith("shard_"):
                return path.parent.parent
            return path.parent

        def _resolve_dataframe_images(self):
            if self.image_key not in self.dataframe.columns:
                return
            root = self._data_root_for_file(self.parquet_files[0])
            rows = [resolve_relative_images(row, root) for row in self.dataframe.to_dict(orient="records")]
            self.dataframe = pd.DataFrame(rows)

        def _read_parquet_with_resolved_images(self, parquet_file: str | Path) -> pd.DataFrame:
            dataframe = pd.read_parquet(parquet_file)
            if self.image_key not in dataframe.columns:
                return dataframe
            root = self._data_root_for_file(parquet_file)
            rows = [resolve_relative_images(row, root) for row in dataframe.to_dict(orient="records")]
            return pd.DataFrame(rows)

        def _read_files_and_process(self):
            dataframes = [self._read_parquet_with_resolved_images(parquet_file) for parquet_file in self.parquet_files]
            self.dataframe = pd.concat(dataframes, ignore_index=True)

            total = len(self.dataframe)
            print(f"dataset len: {len(self.dataframe)}")

            if self.max_samples > 0 and self.max_samples < total:
                if self.shuffle:
                    rngs_args = (self.seed,) if self.seed is not None else ()
                    rng = np.random.default_rng(*rngs_args)
                    indices = rng.choice(total, size=self.max_samples, replace=False)
                else:
                    indices = np.arange(self.max_samples)
                self.dataframe = self.dataframe.iloc[indices.tolist()].reset_index(drop=True)
                print(f"selected {self.max_samples} random samples out of {total}")

            self.messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()
            if self.tools_key in self.dataframe.columns:
                self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
            else:
                self.tools = None
            self.enable_thinking = resolve_enable_thinking_values(self.dataframe, self.enable_thinking_key)
            self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(self.tokenizer)

        def _process_single_message(self, index, message, full_message, tools=None, enable_thinking=None):
            if message["role"] != "system":
                return super()._process_single_message(index, message, full_message, tools, enable_thinking)

            apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
            if enable_thinking is not None:
                apply_chat_template_kwargs["enable_thinking"] = enable_thinking

            empty_user = {"role": "user", "content": ""}
            with_system = self.tokenizer.apply_chat_template(
                [message, empty_user],
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=False,
                **apply_chat_template_kwargs,
            )
            without_system = self.tokenizer.apply_chat_template(
                [empty_user],
                tools=tools,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=False,
                **apply_chat_template_kwargs,
            )
            input_ids = _as_1d_tensor(with_system)[: -len(_as_1d_tensor(without_system))]
            attention_mask = torch.ones_like(input_ids)
            loss_mask = torch.zeros_like(input_ids)
            return input_ids, loss_mask, attention_mask, {}

        def _build_messages(self, example: dict):
            copied = dict(example)
            copied[self.messages_key] = copy.deepcopy(example.get(self.messages_key, []))
            if self.image_key in example:
                copied[self.image_key] = copy.deepcopy(example[self.image_key])
            if self.video_key in example:
                copied[self.video_key] = copy.deepcopy(example[self.video_key])
            return super()._build_messages(copied)

        def __getitem__(self, item):
            from verl.utils.dataset import multiturn_sft_dataset

            # The upstream text-only preview uses the text tokenizer on already-expanded
            # multimodal messages. Qwen3.5 images must stay on the processor path.
            multiturn_sft_dataset.print_assembled_message.called = True
            return super().__getitem__(item)

        def sanity_check(self, input_ids, messages, tools, enable_thinking):
            # Upstream validation re-renders the full already-expanded multimodal
            # message list. For Qwen3.5, per-turn processor tokenization is the valid path.
            return None
