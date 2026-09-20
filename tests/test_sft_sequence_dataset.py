from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
import pytest

from train.sft_dataset import (
    MultiTurnSFTDataset,
    PackingSequenceSFTDataset,
    resolve_enable_thinking_values,
    resolve_relative_images,
)


requires_verl_dataset = pytest.mark.skipif(
    MultiTurnSFTDataset is None, reason="verl.utils.dataset.multiturn_sft_dataset is unavailable"
)


def test_full_sequence_sft_row_schema_if_data_exists():
    path = Path("datasets/packing/sft_seq_v1_final/val.parquet")
    if not path.exists():
        return
    row = pd.read_parquet(path).iloc[0].to_dict()
    assert row["messages"][0]["role"] == "system"
    user_turns = [m for m in row["messages"] if m["role"] == "user"]
    assistant_turns = [m for m in row["messages"] if m["role"] == "assistant"]
    assert len(user_turns) == len(row["images"])
    assert len(assistant_turns) == len(user_turns)
    assert all("<image>" in m["content"] for m in user_turns)


def test_resolve_relative_images_against_data_root(tmp_path):
    root = tmp_path / "sft_seq"
    image = root / "images" / "train" / "case_step_000.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake")
    row = {"images": ["images/train/case_step_000.png"], "messages": []}

    resolved = resolve_relative_images(row, root)

    assert resolved["images"] == [str(image)]
    assert row["images"] == ["images/train/case_step_000.png"]


def test_resolve_relative_images_handles_verl_image_dicts(tmp_path):
    root = tmp_path / "sft_seq"
    image = root / "images" / "train" / "case_step_000.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake")
    row = {"images": [{"image": "images/train/case_step_000.png"}], "messages": []}

    resolved = resolve_relative_images(row, root)

    assert resolved["images"] == [{"image": str(image)}]
    assert row["images"] == [{"image": "images/train/case_step_000.png"}]


@requires_verl_dataset
def test_packing_sequence_sft_dataset_resolves_dataframe_images(tmp_path):
    root = tmp_path / "sft_seq"
    image = root / "images" / "train" / "case_step_000.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake")
    parquet = root / "shard_00000_of_00001" / "train.parquet"
    parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "messages": [{"role": "user", "content": "<image>\nstate"}],
                "images": [{"image": "images/train/case_step_000.png"}],
            }
        ]
    ).to_parquet(parquet)

    dataset = object.__new__(PackingSequenceSFTDataset)
    dataset.parquet_files = [str(parquet)]
    dataset.messages_key = "messages"
    dataset.image_key = "images"
    dataset.tools_key = "tools"
    dataset.enable_thinking_key = "enable_thinking"
    dataset.max_samples = -1
    dataset.shuffle = False
    dataset.seed = None
    dataset.config = OmegaConf.create({"data_root": str(root)})

    dataset.dataframe = pd.read_parquet(parquet)
    dataset._resolve_dataframe_images()

    row = dataset.dataframe.iloc[0].to_dict()
    assert row["images"] == [{"image": str(image)}]


def test_resolve_enable_thinking_defaults_to_false_when_column_missing():
    dataframe = pd.DataFrame([{"messages": []}, {"messages": []}])

    assert resolve_enable_thinking_values(dataframe, "enable_thinking") == [False, False]


def test_resolve_enable_thinking_uses_explicit_column():
    dataframe = pd.DataFrame([{"enable_thinking": False}, {"enable_thinking": True}])

    assert resolve_enable_thinking_values(dataframe, "enable_thinking") == [False, True]


@requires_verl_dataset
def test_packing_sequence_sft_dataset_infers_packing_root(tmp_path):
    root = tmp_path / "packing" / "train_data_sft"
    image = root / "images" / "train" / "easy" / "shard_00000_of_00002" / "case_step_000.png"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake")
    parquet = root / "train" / "easy" / "shard_00000_of_00002" / "train.parquet"
    parquet.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "messages": [{"role": "user", "content": "<image>\nstate"}],
                "images": [{"image": "images/train/easy/shard_00000_of_00002/case_step_000.png"}],
            }
        ]
    ).to_parquet(parquet)

    dataset = object.__new__(PackingSequenceSFTDataset)
    dataset.parquet_files = [str(parquet)]
    dataset.image_key = "images"
    dataset.config = OmegaConf.create({})

    dataframe = dataset._read_parquet_with_resolved_images(parquet)

    assert dataframe.iloc[0]["images"] == [{"image": str(image)}]


@requires_verl_dataset
def test_packing_sequence_sft_dataset_resolves_each_parquet_against_its_own_root(tmp_path):
    first_root = tmp_path / "first_seq"
    second_root = tmp_path / "second_seq"
    first_image = first_root / "images" / "train" / "case_step_000.png"
    second_image = second_root / "images" / "train" / "case_step_000.png"
    first_image.parent.mkdir(parents=True)
    second_image.parent.mkdir(parents=True)
    first_image.write_bytes(b"first")
    second_image.write_bytes(b"second")

    first_parquet = first_root / "shard_00000_of_00002" / "train.parquet"
    second_parquet = second_root / "shard_00001_of_00002" / "train.parquet"
    first_parquet.parent.mkdir(parents=True)
    second_parquet.parent.mkdir(parents=True)
    row = {
        "messages": [{"role": "user", "content": "<image>\nstate"}],
        "images": [{"image": "images/train/case_step_000.png"}],
    }
    pd.DataFrame([row]).to_parquet(first_parquet)
    pd.DataFrame([row]).to_parquet(second_parquet)

    dataset = object.__new__(PackingSequenceSFTDataset)
    dataset.parquet_files = [str(first_parquet), str(second_parquet)]
    dataset.image_key = "images"
    dataset.config = OmegaConf.create({})

    dataframes = [dataset._read_parquet_with_resolved_images(path) for path in dataset.parquet_files]

    images = [dataframe.iloc[0]["images"][0]["image"] for dataframe in dataframes]
    assert images == [str(first_image), str(second_image)]


@requires_verl_dataset
def test_packing_sequence_sft_dataset_build_messages_does_not_mutate_rows(monkeypatch):
    from verl.utils.dataset import multiturn_sft_dataset

    dataset = object.__new__(PackingSequenceSFTDataset)
    dataset.messages_key = "messages"
    dataset.image_key = "images"
    dataset.video_key = "videos"
    dataset.image_patch_size = 14
    dataset.processor = object()
    monkeypatch.setattr(multiturn_sft_dataset, "process_image", lambda image, image_patch_size: image)

    example = {
        "messages": [{"role": "user", "content": "<image>\nstate"}],
        "images": ["image_a.png"],
    }

    first = dataset._build_messages(example)
    second = dataset._build_messages(example)

    assert example["messages"][0]["content"] == "<image>\nstate"
    assert first[0]["content"][0] == {"type": "image", "image": "image_a.png"}
    assert second[0]["content"][0] == {"type": "image", "image": "image_a.png"}
