from __future__ import annotations

import argparse
import glob
import os
import subprocess
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import ListConfig, OmegaConf


def build_verl_command(cfg) -> list[str]:
    if cfg.task_type != "sft":
        raise ValueError(f"invalid task_type: {cfg.task_type}")
    return ["python", "-m", "train.run_verl_module", "verl.trainer.sft_trainer"]


def expand_train_files(train_dir: str, pattern: str) -> list[str]:
    train_pattern = str(Path(train_dir) / pattern)
    paths = sorted(glob.glob(train_pattern, recursive=True))
    if not paths:
        if os.environ.get("DRY_RUN") == "1" or os.environ.get("PACKLAB_ALLOW_MISSING_TRAIN_FILES") == "1":
            return [train_pattern]
        raise FileNotFoundError(f"no train files match: {train_pattern}")
    return paths


def _list_literal(items: list[str]) -> str:
    return "[" + ",".join(repr(str(item)) for item in items) + "]"


def _file_literal(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, list | tuple | ListConfig):
        return _list_literal([str(item) for item in value])
    return str(value)


def _bool(value) -> str:
    return "True" if bool(value) else "False"


def _env_override(name: str, current):
    return os.environ.get(name, current)


def _env_list_override(name: str, current):
    raw = os.environ.get(name)
    if raw is None:
        return current
    return [item for item in raw.split(os.pathsep) if item]


def _select(cfg, key: str, default=None):
    return OmegaConf.select(cfg, key, default=default)


def load_config(config_path: str):
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    parts = list(path.parts)
    if "configs" in parts:
        idx = parts.index("configs")
        config_root = Path(*parts[: idx + 1])
        config_name = str(Path(*parts[idx + 1 :]))
    else:
        config_root = path.parent
        config_name = path.name
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        cfg = compose(config_name=config_name)
    top_key = Path(config_name).parts[0]
    if top_key in cfg and "task_type" not in cfg:
        return cfg[top_key]
    return cfg


def sft_overrides(cfg) -> list[str]:
    train_dir = _env_override("SFT_TRAIN_DIR", cfg.data.train_dir)
    train_glob = _env_override("SFT_TRAIN_GLOB", cfg.data.train_glob)
    train_files = expand_train_files(train_dir, train_glob)
    val_files = _env_list_override("SFT_VAL_FILES", cfg.data.val_file)
    data_root = _env_override("SFT_DATA_ROOT", _select(cfg, "data.data_root"))
    model_path = _env_override("MODEL_PATH", cfg.model.get("base_path", cfg.model.get("path", "")))
    output_dir = _env_override("OUTPUT_DIR", cfg.checkpoint.output_dir)
    nnodes = _env_override("NNODES", cfg.cluster.nnodes)
    nproc = _env_override("NPROC_PER_NODE", cfg.cluster.nproc_per_node)
    overrides = [
        f"data.train_files={_list_literal(train_files)}",
        f"data.val_files={_file_literal(val_files)}",
        f"data.messages_key={cfg.data.messages_key}",
        f"data.max_length={cfg.data.max_length}",
        f"data.max_token_len_per_gpu={_select(cfg, 'data.max_token_len_per_gpu', cfg.data.max_length)}",
        f"data.truncation={cfg.data.truncation}",
        f"data.train_batch_size={cfg.optimization.train_batch_size}",
        f"data.micro_batch_size_per_gpu={cfg.optimization.micro_batch_size_per_gpu}",
        f"data.use_dynamic_bsz={_bool(_select(cfg, 'data.use_dynamic_bsz', True)).lower()}",
        f"data.num_workers={_select(cfg, 'data.num_workers', 8)}",
        f"optim.lr={cfg.optimization.lr}",
        f"model.path={model_path}",
        f"model.use_remove_padding={_bool(_select(cfg, 'model_runtime.use_remove_padding', False)).lower()}",
        f"model.enable_gradient_checkpointing={_bool(_select(cfg, 'model_runtime.enable_gradient_checkpointing', True)).lower()}",
        f"model.enable_activation_offload={_bool(_select(cfg, 'model_runtime.enable_activation_offload', False)).lower()}",
        f"engine.model_dtype={_select(cfg, 'fsdp.model_dtype', 'bf16')}",
        f"engine.use_torch_compile={_bool(_select(cfg, 'fsdp.use_torch_compile', False)).lower()}",
        f"engine.param_offload={_bool(_select(cfg, 'fsdp.param_offload', True)).lower()}",
        f"engine.optimizer_offload={_bool(_select(cfg, 'fsdp.optimizer_offload', True)).lower()}",
        f"trainer.total_epochs={cfg.optimization.epochs}",
        f"trainer.default_local_dir={output_dir}",
        f"trainer.nnodes={nnodes}",
        f"trainer.n_gpus_per_node={nproc}",
    ]
    max_ckpt_to_keep = _select(cfg, "trainer.max_ckpt_to_keep")
    if max_ckpt_to_keep is not None:
        overrides.append(f"trainer.max_ckpt_to_keep={max_ckpt_to_keep}")
    for key, value in _select(cfg, "model_runtime.override_config", {}).items():
        overrides.append(f"+model.override_config.{key}={value}")
    if _select(cfg, "data.custom_cls.path"):
        overrides.extend([
            f"data.custom_cls.path={cfg.data.custom_cls.path}",
            f"data.custom_cls.name={cfg.data.custom_cls.name}",
        ])
    if data_root:
        overrides.append(f"+data.data_root={data_root}")
    return overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--print-cmd", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cmd = build_verl_command(cfg) + sft_overrides(cfg) + args.overrides
    print(" ".join(cmd), flush=True)
    if args.print_cmd:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
