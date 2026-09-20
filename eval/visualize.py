from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from eval.recorder import EpisodeRecorder
from eval.vllm_runner import _load_env_cfg
from phy_env.env import PackingEnv


def load_action_record(path: str | Path, case_id: str | None = None) -> dict:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = str(record.get("eval_sample_id") or record.get("case_id") or "")
            if case_id is None or record_id == case_id:
                return record
    raise KeyError(f"case id not found: {case_id}")


def _sample_from_record(record: dict) -> dict:
    return {
        "container_size_cm": record["container_size_cm"],
        "buffer_size": int(record["buffer_size"]),
        "object_sequence": record["object_sequence"],
        "seed": record.get("seed"),
    }


def _action_for_step(step: dict) -> dict:
    action = step.get("action")
    if not isinstance(action, dict):
        raise ValueError(f"step missing action: {step}")
    return {
        "object_id": int(action["object_id"]),
        "x": int(action["x"]),
        "y": int(action["y"]),
        "rotation": int(action.get("rotation", 0)),
    }


def replay(record: dict, output_dir: str | Path, env_config: str | Path, save_video: bool = True) -> dict:
    env_cfg = _load_env_cfg(env_config)
    env_cfg = OmegaConf.create(OmegaConf.to_container(env_cfg, resolve=True))
    env_cfg.env.render = False
    env_cfg.env.render_aa_scale = max(2, int(env_cfg.env.get("render_aa_scale", 1)))
    env = PackingEnv(env_cfg)
    recorder = None
    executed = []
    try:
        env.reset(_sample_from_record(record))
        recorder = EpisodeRecorder(env, output_dir, save_video=save_video)
        recorder.begin()
        for step_index, step in enumerate(record.get("actions", [])):
            action = _action_for_step(step)
            recorder.before_step()
            result = env.step_relaxed(
                action,
                force_outside=bool(step.get("forced_outside", False)),
                outside_seed=step.get("forced_seed"),
            )
            recorder.after_step()
            executed.append(
                {
                    "step_index": step_index,
                    "action": action,
                    "valid": bool(result.info.get("valid", False)),
                    "reason": result.info.get("reason"),
                    "done": bool(result.done),
                }
            )
            if result.done:
                break
        summary = {
            "case_id": str(record.get("eval_sample_id") or record.get("case_id") or ""),
            "executed_steps": len(executed),
            "termination_reason": env.termination_reason,
            "output_dir": str(output_dir),
        }
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    finally:
        if recorder is not None:
            recorder.finish()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a PackLab action sequence into images and video.")
    parser.add_argument("--action-sequences", required=True)
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--output-dir", default="outputs/visualization")
    parser.add_argument("--env-config", default="configs/phy_env/env.yaml")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    record = load_action_record(args.action_sequences, case_id=args.case_id)
    summary = replay(record, args.output_dir, args.env_config, save_video=not args.no_video)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
