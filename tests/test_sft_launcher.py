from pathlib import Path

import pytest
from omegaconf import OmegaConf

from train.launch_verl import build_verl_command, expand_train_files, load_config, sft_overrides


def test_sft_launcher_uses_sft_entrypoint():
    cfg = OmegaConf.create({"task_type": "sft"})

    assert build_verl_command(cfg) == ["python", "-m", "train.run_verl_module", "verl.trainer.sft_trainer"]


def test_expand_train_files_handles_recursive_glob(tmp_path):
    (tmp_path / "easy" / "shard_00000_of_00002").mkdir(parents=True)
    (tmp_path / "hard" / "shard_00001_of_00002").mkdir(parents=True)
    (tmp_path / "easy" / "shard_00000_of_00002" / "train.parquet").write_text("")
    (tmp_path / "hard" / "shard_00001_of_00002" / "train.parquet").write_text("")

    files = expand_train_files(str(tmp_path), "**/*.parquet")

    assert files == [
        str(tmp_path / "easy" / "shard_00000_of_00002" / "train.parquet"),
        str(tmp_path / "hard" / "shard_00001_of_00002" / "train.parquet"),
    ]


def test_expand_train_files_allows_missing_files_for_dry_run(tmp_path, monkeypatch):
    monkeypatch.setenv("DRY_RUN", "1")

    assert expand_train_files(str(tmp_path), "*/train.parquet") == [str(tmp_path / "*/train.parquet")]


def test_expand_train_files_requires_existing_files_for_real_run(tmp_path, monkeypatch):
    monkeypatch.delenv("DRY_RUN", raising=False)
    monkeypatch.delenv("PACKLAB_ALLOW_MISSING_TRAIN_FILES", raising=False)

    with pytest.raises(FileNotFoundError):
        expand_train_files(str(tmp_path), "*/train.parquet")


def test_sft_overrides_include_data_model_and_checkpoint_paths(tmp_path):
    (tmp_path / "shard_00000_of_00100").mkdir()
    (tmp_path / "shard_00000_of_00100" / "train.parquet").write_text("")
    cfg = OmegaConf.create({
        "task_type": "sft",
        "model": {"base_path": "Qwen/Qwen3.5-9B"},
        "cluster": {"nnodes": 1, "nproc_per_node": 8},
        "data": {
            "train_dir": str(tmp_path),
            "train_glob": "shard_*_of_00100/train.parquet",
            "val_file": "val.parquet",
            "data_root": "data/PackData-20K",
            "messages_key": "messages",
            "image_key": "images",
            "custom_cls": {"path": "train/sft_dataset.py", "name": "PackingSequenceSFTDataset"},
            "max_length": 4096,
            "truncation": "error",
        },
        "optimization": {"train_batch_size": 4, "micro_batch_size_per_gpu": 1, "epochs": 1, "lr": 1e-6},
        "model_runtime": {"use_remove_padding": False, "enable_gradient_checkpointing": True},
        "fsdp": {"model_dtype": "bf16", "use_torch_compile": False, "param_offload": True, "optimizer_offload": True},
        "checkpoint": {"output_dir": "outputs/checkpoints/sft"},
        "trainer": {"max_ckpt_to_keep": 2},
    })

    overrides = sft_overrides(cfg)

    assert "data.train_files=['" + str(tmp_path / "shard_00000_of_00100" / "train.parquet") + "']" in overrides
    assert "data.val_files=val.parquet" in overrides
    assert "model.path=Qwen/Qwen3.5-9B" in overrides
    assert "trainer.default_local_dir=outputs/checkpoints/sft" in overrides
    assert "data.custom_cls.path=train/sft_dataset.py" in overrides
    assert "data.custom_cls.name=PackingSequenceSFTDataset" in overrides
    assert "+data.data_root=data/PackData-20K" in overrides


def test_public_training_config_uses_release_assets():
    cfg = load_config("configs/train/packing_sft.yaml")

    assert cfg.task_type == "sft"
    assert cfg.data.train_dir == "data/PackData-20K/train"
    assert cfg.data.val_file is None
    assert cfg.data.data_root == "data/PackData-20K"
    assert cfg.checkpoint.output_dir == "outputs/checkpoints/packlab_vlm_9b_sft"


def test_public_training_config_emits_null_validation_files(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "1")
    cfg = load_config("configs/train/packing_sft.yaml")

    assert "data.val_files=null" in sft_overrides(cfg)


def test_launcher_accepts_sft_only():
    text = Path("train/launch_verl.py").read_text(encoding="utf-8").lower()

    assert "task_type != \"sft\"" in text
    assert "trainer.sft_trainer" in text
