from __future__ import annotations

import builtins
import importlib
import os
import sys
import traceback

from train.verl_compat import apply_verl_compat_patches


def _patch_torch25_dtensor_annotations() -> None:
    try:
        from torch.distributed.tensor import DTensor, Shard
        from torch.distributed.tensor._dtensor_spec import DTensorSpec
    except Exception:
        return
    builtins.DTensor = DTensor
    builtins.Shard = Shard
    builtins.DTensorSpec = DTensorSpec


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m train.run_verl_module <module> [args...]")
    module = sys.argv[1]
    sys.argv = [module] + sys.argv[2:]
    _patch_torch25_dtensor_annotations()
    apply_verl_compat_patches()
    if module == "verl.trainer.sft_trainer":
        _run_sft_trainer_module(module)
        return
    raise SystemExit(f"invalid wrapped verl module: {module}")


def _run_sft_trainer_module(module: str) -> None:
    trainer_module = importlib.import_module(module)
    trainer_module.destroy_global_process_group = lambda: None
    try:
        trainer_module.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 0 if exc.code is None else 1
        if code != 0:
            print(f"wrapped verl module exited with code {exc.code!r}", file=sys.stderr, flush=True)
            raise
    except BaseException:
        traceback.print_exc()
        raise
    else:
        code = 0

    # The SFT trainer can leave NCCL/worker teardown threads alive after the
    # final metrics and checkpoint are written. Exit the rank process directly
    # once Hydra returns successfully so torchrun can reap all ranks.
    os._exit(code)


if __name__ == "__main__":
    main()
