"""Dataset build and replay helpers."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from data_engine.generators.layer_cuboid import GenerationConfig, LayerCuboidGenerator
from data_engine.prompts.renderer import render_user_turn_from_state
import phy_env.components  # noqa: F401
from phy_env.env import PackingEnv
from phy_env.render_utils import concat_images_vertically, concat_step_images, save_rgb_image, save_rgba_image, save_video, to_rgb_uint8
from runtime.messages import build_system_message
from runtime.state import build_container_state

BLANK_HEIGHT_MAP_RGB = np.zeros((224, 224, 3), dtype=np.uint8)


def episode_to_eval_row(episode: dict, split: str, index: int, prompt_template_id: str = "packing_step_v1") -> dict:
    return {
        "data_source": "packing_eval",
        "prompt_template_id": prompt_template_id,
        "ability": "spatial_packing",
        "extra_info": {
            "split": split,
            "index": index,
            "difficulty": episode["difficulty"],
            "generator": episode["generator"],
            "gt_plan": episode["gt_plan"],
            "interaction_kwargs": {
                "name": "packing",
                "sample_id": episode["sample_id"],
                "container_size_cm": episode["container_size_cm"],
                "buffer_size": episode["buffer_size"],
                "object_sequence": episode["object_sequence"],
                "seed": episode["seed"],
            },
        },
    }


def build_sft_rows(
    episodes: list[dict],
    env_cfg,
    image_dir: Path,
    split: str,
    prompt_cfg,
    prompt_template_id: str,
    prompt_view: dict | None = None,
) -> tuple[list[dict], dict]:
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    report = {"episodes": len(episodes), "failed": 0, "completed": 0, "score_non_1": 0, "final_scores": []}
    for episode in episodes:
        try:
            materialized_rows, ep_report = _replay_episode_to_sft_materialized_rows(
                episode,
                env_cfg,
                image_dir,
                split,
                prompt_cfg,
                prompt_template_id,
                prompt_view=prompt_view,
            )
            rows.extend(materialized_rows)
            report["completed"] += 1
            report["final_scores"].append(ep_report["final_score"])
            if ep_report["final_score"] != 1.0:
                report["score_non_1"] += 1
        except Exception as exc:
            report["failed"] += 1
            report.setdefault("errors", []).append({"episode_id": episode.get("episode_id"), "error": repr(exc)})
    return rows, report


def _build_sft_rows_with_replacement(
    generator: LayerCuboidGenerator,
    difficulty: str,
    target_count: int,
    shard_index: int,
    num_shards: int,
    env_cfg,
    image_dir: Path,
    split: str,
    prompt_cfg,
    prompt_template_id: str,
    prompt_view: dict | None = None,
) -> tuple[list[dict], dict, list[dict]]:
    """Build exactly target_count replay-clean rows, replacing rejected candidates."""
    rows = []
    accepted_episodes = []
    report = {
        "episodes": int(target_count),
        "completed": 0,
        "failed": 0,
        "score_non_1": 0,
        "attempted": 0,
        "rejected": 0,
        "final_scores": [],
    }
    max_attempts = max(int(target_count) * 50, int(target_count) + 100)
    attempt = 0
    while len(accepted_episodes) < target_count and attempt < max_attempts:
        candidate_index = shard_index + attempt * num_shards
        attempt += 1
        episode = generator.generate_episode(
            candidate_index,
            difficulty=difficulty,
            buffer_size=generator.cfg.buffer_sizes[candidate_index % len(generator.cfg.buffer_sizes)],
        )
        report["attempted"] += 1
        try:
            materialized_rows, ep_report = _replay_episode_to_sft_materialized_rows(
                episode,
                env_cfg,
                image_dir,
                split,
                prompt_cfg,
                prompt_template_id,
                prompt_view=prompt_view,
            )
        except Exception as exc:
            report["rejected"] += 1
            report.setdefault("rejected_errors", []).append({"episode_id": episode.get("episode_id"), "error": repr(exc)})
            _remove_episode_images(image_dir, episode["episode_id"])
            continue
        if ep_report["final_score"] != 1.0:
            report["rejected"] += 1
            report.setdefault("rejected_errors", []).append({
                "episode_id": episode.get("episode_id"),
                "error": f"final_score={ep_report['final_score']}",
            })
            _remove_episode_images(image_dir, episode["episode_id"])
            continue
        rows.extend(materialized_rows)
        accepted_episodes.append(episode)
        report["completed"] += 1
        report["final_scores"].append(ep_report["final_score"])
    if len(accepted_episodes) != target_count:
        raise RuntimeError(
            f"only built {len(accepted_episodes)} valid SFT episodes for {difficulty}; "
            f"target={target_count}, attempted={report['attempted']}"
        )
    return rows, report, accepted_episodes


def _remove_episode_images(image_dir: Path, episode_id: str) -> None:
    for image_path in Path(image_dir).glob(f"{episode_id}_step_*.png"):
        image_path.unlink(missing_ok=True)


def build_eval_dataset(cfg) -> dict:
    out_dir = Path(cfg.output_dir)
    manifest = {"mode": "eval", "split": "test", "difficulties": {}}
    all_episodes = []
    for difficulty, diff_cfg in _iter_difficulty_generation_cfgs(cfg.generation):
        episodes = LayerCuboidGenerator(_generation_config_from_cfg(diff_cfg)).generate()
        rows = [
            episode_to_eval_row(ep, split="test", index=i, prompt_template_id=cfg.prompt_template_id)
            for i, ep in enumerate(episodes)
        ]
        parquet_path = out_dir / difficulty / "test.parquet"
        write_parquet(rows, parquet_path)
        manifest["difficulties"][difficulty] = _dataset_manifest(
            episodes,
            split="test",
            mode="eval",
            parquet_path=str(parquet_path),
        )
        all_episodes.extend(episodes)
    manifest["summary"] = _dataset_manifest(all_episodes, split="test", mode="eval", parquet_path=str(out_dir))
    write_json(manifest, out_dir / "metadata" / "manifest.json")
    return manifest


def build_sft_dataset(cfg, env_cfg) -> dict:
    out_dir = Path(cfg.output_dir)
    if bool(cfg.get("timestamped_output", False)):
        out_dir = out_dir / time.strftime("%Y%m%d_%H%M%S")
    split = str(cfg.get("split", "train"))
    shard_index = int(cfg.get("shard_index", 0))
    num_shards = int(cfg.get("num_shards", 1))
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    prompt_cfg = _load_cfg(cfg.prompt_config_path)
    prompt_view = _prompt_view_from_cfg(cfg)
    shard_name = f"shard_{shard_index:05d}_of_{num_shards:05d}"
    manifest = {"mode": "sft", "split": split, "shard_index": shard_index, "num_shards": num_shards, "difficulties": {}, "prompt_view": prompt_view}
    total_failed = 0
    total_score_non_1 = 0
    for difficulty, diff_cfg in _iter_difficulty_generation_cfgs(cfg.generation):
        gen_cfg = _generation_config_from_cfg(diff_cfg)
        generator = LayerCuboidGenerator(gen_cfg)
        target_count = sum(1 for idx in range(gen_cfg.num_episodes) if idx % num_shards == shard_index)
        shard_dir = out_dir / split / difficulty / shard_name
        rows, report, episodes = _build_sft_rows_with_replacement(
            generator=generator,
            difficulty=difficulty,
            target_count=target_count,
            shard_index=shard_index,
            num_shards=num_shards,
            env_cfg=env_cfg,
            image_dir=out_dir / "images" / split / difficulty / shard_name,
            split=split,
            prompt_cfg=prompt_cfg,
            prompt_template_id=cfg.prompt_template_id,
            prompt_view=prompt_view,
        )
        parquet_path = shard_dir / "train.parquet"
        write_jsonl(
            _unique_canonical_trajectories(rows),
            out_dir / "canonical" / split / difficulty / shard_name / "trajectories.jsonl",
        )
        write_parquet([_without_canonical(row) for row in rows], parquet_path)
        diff_manifest = _dataset_manifest(episodes, split=split, mode="sft", parquet_path=str(parquet_path))
        diff_manifest["shard_index"] = shard_index
        diff_manifest["num_shards"] = num_shards
        diff_manifest["sequence_rows"] = len(rows)
        diff_manifest["replay_report"] = report
        manifest["difficulties"][difficulty] = diff_manifest
        total_failed += int(report["failed"])
        total_score_non_1 += int(report["score_non_1"])
        write_json(diff_manifest, shard_dir / "manifest.json")
        if bool(cfg.get("write_check_report", False)):
            write_json(_sequence_check_report(rows, out_dir, report), shard_dir / "check_report.json")
    write_json(manifest, out_dir / "metadata" / f"{shard_name}_manifest.json")
    if bool(cfg.get("write_check_report", False)):
        _write_sequence_summary(out_dir, expected_num_shards=num_shards)
    if total_failed:
        raise RuntimeError(f"SFT replay failed for {total_failed} episodes")
    if total_score_non_1:
        raise RuntimeError(f"SFT replay produced {total_score_non_1} non-1 scores")
    return manifest


def _sft_shard_is_complete(out_dir: Path, shard_index: int, num_shards: int, difficulties: list[str]) -> bool:
    shard_name = f"shard_{shard_index:05d}_of_{num_shards:05d}"
    for difficulty in difficulties:
        manifest_path = out_dir / "train" / difficulty / shard_name / "manifest.json"
        if not manifest_path.exists():
            return False
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        replay_report = manifest.get("replay_report", {})
        if int(replay_report.get("failed", 0)) or int(replay_report.get("score_non_1", 0)):
            return False
    return True


def build_sft_shards(
    cfg,
    env_cfg,
    start_shard: int | None = None,
    end_shard: int | None = None,
    num_shards: int | None = None,
    skip_completed: bool | None = None,
    summarize: bool = True,
) -> list[dict]:
    num_shards = int(num_shards if num_shards is not None else cfg.get("num_shards", 1))
    start_shard = int(start_shard if start_shard is not None else cfg.get("start_shard", 0))
    end_shard = int(end_shard if end_shard is not None else cfg.get("end_shard", num_shards - 1))
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if start_shard < 0 or end_shard >= num_shards or start_shard > end_shard:
        raise ValueError("shard range must satisfy 0 <= start_shard <= end_shard < num_shards")

    skip_completed = bool(skip_completed if skip_completed is not None else cfg.get("skip_completed", False))
    out_dir = Path(cfg.output_dir)
    difficulties = [difficulty for difficulty, _ in _iter_difficulty_generation_cfgs(cfg.generation)]
    manifests = []
    for shard_index in range(start_shard, end_shard + 1):
        if skip_completed and _sft_shard_is_complete(out_dir, shard_index, num_shards, difficulties):
            print(f"skip completed shard {shard_index}/{num_shards}")
            continue
        shard_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        shard_cfg.shard_index = shard_index
        shard_cfg.num_shards = num_shards
        manifests.append(build_sft_dataset(shard_cfg, env_cfg))
    if summarize:
        _write_sequence_summary(out_dir, expected_num_shards=num_shards)
    return manifests


def build_sft_val_dataset(cfg, env_cfg) -> dict:
    out_dir = Path(cfg.output_dir)
    split = str(cfg.get("split", "val"))
    prompt_cfg = _load_cfg(cfg.prompt_config_path)
    prompt_view = _prompt_view_from_cfg(cfg)
    generation_cfg = cfg.get("val_generation", cfg.generation)
    manifest = {"mode": "sft", "split": split, "difficulties": {}, "prompt_view": prompt_view}
    all_episodes = []
    total_failed = 0
    total_score_non_1 = 0
    for difficulty, diff_cfg in _iter_difficulty_generation_cfgs(generation_cfg):
        gen_cfg = _generation_config_from_cfg(diff_cfg)
        rows, report, episodes = _build_sft_rows_with_replacement(
            generator=LayerCuboidGenerator(gen_cfg),
            difficulty=difficulty,
            target_count=int(gen_cfg.num_episodes),
            shard_index=0,
            num_shards=1,
            env_cfg=env_cfg,
            image_dir=out_dir / "images" / split / difficulty,
            split=split,
            prompt_cfg=prompt_cfg,
            prompt_template_id=cfg.prompt_template_id,
            prompt_view=prompt_view,
        )
        parquet_path = out_dir / split / difficulty / "val.parquet"
        write_jsonl(_unique_canonical_trajectories(rows), out_dir / "canonical" / split / difficulty / "trajectories.jsonl")
        write_parquet([_without_canonical(row) for row in rows], parquet_path)
        diff_manifest = _dataset_manifest(episodes, split=split, mode="sft", parquet_path=str(parquet_path))
        diff_manifest["sequence_rows"] = len(rows)
        diff_manifest["replay_report"] = report
        manifest["difficulties"][difficulty] = diff_manifest
        total_failed += int(report["failed"])
        total_score_non_1 += int(report["score_non_1"])
        all_episodes.extend(episodes)
    manifest["summary"] = _dataset_manifest(all_episodes, split=split, mode="sft", parquet_path=str(out_dir / split))
    write_json(manifest, out_dir / f"{split}_manifest.json")
    if total_failed:
        raise RuntimeError(f"SFT val replay failed for {total_failed} episodes")
    if total_score_non_1:
        raise RuntimeError(f"SFT val replay produced {total_score_non_1} non-1 scores")
    return manifest


def replay_episode_with_visuals(episode: dict, env_cfg, out_dir: Path) -> dict:
    env = PackingEnv(env_cfg)
    report = {
        "episode_id": episode["episode_id"],
        "passed": False,
        "valid_steps": 0,
        "errors": [],
        "final_score": None,
        "termination_reason": None,
        "visualization_dir": str(out_dir),
    }
    sample = {
        "container_size_cm": episode["container_size_cm"],
        "buffer_size": episode["buffer_size"],
        "object_sequence": episode["object_sequence"],
        "seed": episode["seed"],
    }
    try:
        obs = env.reset(sample)
        subdirs = {
            "buffer": out_dir / "before_placement",
            "place": out_dir / "after_placement",
            "height": out_dir / "height_map",
            "overlay": out_dir / "overlays",
            "summary": out_dir / "summaries",
        }
        for path in subdirs.values():
            path.mkdir(parents=True, exist_ok=True)
        frames = []
        summaries = []
        init_frame = env.capture_scene_rgb()
        save_rgb_image(subdirs["place"] / "after_placement_step_000.png", init_frame)
        frames.append(init_frame)
        for step_index, gt in enumerate(episode["gt_plan"]):
            object_id = _find_visible_object_id(obs.buffer_objects, gt["internal_id"])
            action = dict(gt["action"])
            action["object_id"] = object_id
            before_frame = env.capture_scene_rgb()
            save_rgb_image(subdirs["buffer"] / f"before_placement_step_{step_index + 1:03d}.png", before_frame)
            result = env.step(action)
            after_frame = env.capture_scene_rgb()
            save_rgb_image(subdirs["place"] / f"after_placement_step_{step_index + 1:03d}.png", after_frame)
            height_rgba = env.colored_height_map_image()
            save_rgba_image(subdirs["height"] / f"height_map_step_{step_index + 1:03d}.png", height_rgba)
            height_rgb = to_rgb_uint8((height_rgba * 255.0).clip(0, 255).astype("uint8"))
            overlay = _draw_gt_overlay(height_rgb, gt["target_box_cells"], env.container_size)
            save_rgb_image(subdirs["overlay"] / f"overlay_step_{step_index + 1:03d}.png", overlay)
            summary = concat_step_images(before_frame, after_frame, height_rgb, step_index + 1)
            save_rgb_image(subdirs["summary"] / f"summary_step_{step_index + 1:03d}.png", summary)
            summaries.append(summary)
            frames.extend([before_frame, after_frame])
            if not result.info["valid"]:
                report["errors"].append({"step": step_index, "reason": result.info["reason"], "action": action})
                break
            report["valid_steps"] += 1
            obs = result.observation
        if frames:
            save_video(out_dir / "placement_keyframes.mp4", frames, fps=4)
        if summaries:
            save_rgb_image(subdirs["summary"] / "summary_all_steps.png", concat_images_vertically(summaries))
        report["final_score"] = float(env.get_final_score())
        report["termination_reason"] = env.termination_reason
        report["passed"] = (
            report["valid_steps"] == len(episode["gt_plan"])
            and env.termination_reason == "completed"
            and report["final_score"] > 0.99
        )
    except Exception as exc:
        report["errors"].append({"error": repr(exc)})
    finally:
        env.close()
    return report


def write_parquet(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


def write_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(rows: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _generation_config_from_cfg(cfg) -> GenerationConfig:
    return GenerationConfig(
        num_episodes=int(cfg.num_episodes),
        difficulties=dict(cfg.difficulties),
        buffer_sizes=[int(v) for v in cfg.buffer_sizes],
        seed=int(cfg.seed),
        first_anchor_distribution=dict(cfg.get("first_anchor_distribution", {})),
        order_strategy=str(cfg.get("order_strategy", "layer_random_anchor")),
    )


def _single_difficulty_name(difficulties: dict) -> str:
    active = [str(k) for k, v in dict(difficulties).items() if float(v) > 0]
    return active[0] if len(active) == 1 else "mixed"


def _iter_difficulty_generation_cfgs(cfg):
    per_difficulty = cfg.get("per_difficulty", None)
    if per_difficulty is not None:
        for difficulty, values in per_difficulty.items():
            yield str(difficulty), values
        return
    yield _single_difficulty_name(cfg.difficulties), cfg


def _prompt_view_from_cfg(cfg) -> dict:
    raw = cfg.get("prompt_view", {})
    return {
        "history_mode": str(raw.get("history_mode", "full")),
        "height_map_mode": str(raw.get("height_map_mode", "image")),
    }


def config_to_container(cfg) -> str:
    return OmegaConf.to_yaml(cfg)


def _dataset_manifest(episodes: list[dict], split: str, mode: str, parquet_path: str) -> dict:
    difficulty_counts = {}
    buffer_counts = {}
    for ep in episodes:
        difficulty_counts[ep["difficulty"]] = difficulty_counts.get(ep["difficulty"], 0) + 1
        buffer_key = str(ep["buffer_size"])
        buffer_counts[buffer_key] = buffer_counts.get(buffer_key, 0) + 1
    return {
        "mode": mode,
        "split": split,
        "num_episodes": len(episodes),
        "parquet_path": parquet_path,
        "difficulty_counts": difficulty_counts,
        "buffer_size_counts": buffer_counts,
        "seed_min": min(ep["seed"] for ep in episodes) if episodes else None,
        "seed_max": max(ep["seed"] for ep in episodes) if episodes else None,
    }


def _replay_episode_to_sft_materialized_rows(
    episode: dict,
    env_cfg,
    image_dir: Path,
    split: str,
    prompt_cfg,
    prompt_template_id: str,
    prompt_view: dict | None = None,
) -> tuple[list[dict], dict]:
    trajectory, ep_report = replay_episode_to_canonical_trajectory(episode, env_cfg, image_dir)
    return materialize_sft_rows(trajectory, split, prompt_cfg, prompt_template_id, prompt_view=prompt_view), ep_report


def replay_episode_to_canonical_trajectory(episode: dict, env_cfg, image_dir: Path) -> tuple[dict, dict]:
    """Replay an expert episode into prompt-agnostic states, images, and actions."""
    env = PackingEnv(env_cfg)
    sample = {
        "container_size_cm": episode["container_size_cm"],
        "buffer_size": episode["buffer_size"],
        "object_sequence": episode["object_sequence"],
        "seed": episode["seed"],
    }
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    steps = []
    valid_steps = 0
    try:
        obs = env.reset(sample)
        container_state = build_container_state(env, sample["container_size_cm"])
        for step_index, gt in enumerate(episode["gt_plan"]):
            visible = obs.buffer_objects
            state = _state_from_obs(episode, env, visible)
            image_rel_path = _image_rel_dir(image_dir) / f"{episode['episode_id']}_step_{step_index:03d}.png"
            image_path = image_dir / image_rel_path.name
            Image.fromarray(obs.image).save(image_path)

            object_id = _find_visible_object_id(visible, gt["internal_id"])
            action = dict(gt["action"])
            action["object_id"] = object_id
            steps.append({
                "step_index": step_index,
                "state": state,
                "assistant_action": action,
                "image_path": str(image_path.resolve()),
                "image_rel_path": str(image_rel_path),
                "gt_internal_id": gt["internal_id"],
            })

            result = env.step(action)
            if not result.info["valid"]:
                raise RuntimeError(f"invalid GT action at step {step_index}: {result.info['reason']}")
            valid_steps += 1
            obs = result.observation

        final_score = float(env.get_final_score())
        if abs(final_score - 1.0) < 1e-9:
            final_score = 1.0
        ep_report = {
            "episode_id": episode["episode_id"],
            "valid_steps": valid_steps,
            "final_score": final_score,
            "termination_reason": env.termination_reason,
            "passed": valid_steps == len(episode["gt_plan"]) and env.termination_reason == "completed" and final_score == 1.0,
        }
        if not ep_report["passed"]:
            raise RuntimeError(f"sequence replay did not complete cleanly: {ep_report}")
        return {
            "schema_version": "packing_canonical_trajectory_v1",
            "data_source": "packing_canonical_trajectory",
            "prompt_template_id": None,
            "episode_id": episode["episode_id"],
            "difficulty": episode["difficulty"],
            "buffer_size": episode["buffer_size"],
            "container_bucket": episode["container_bucket"],
            "generator": episode["generator"],
            "container_state": container_state,
            "container_size_cm": episode["container_size_cm"],
            "container_grid_size": episode["container_grid_size"],
            "grid_cell_size_cm": episode["grid_cell_size_cm"],
            "object_sequence": episode["object_sequence"],
            "seed": episode["seed"],
            "gt_plan": episode["gt_plan"],
            "steps": steps,
            "final_score": final_score,
            "termination_reason": env.termination_reason,
            "valid_steps": valid_steps,
        }, ep_report
    finally:
        env.close()


def materialize_sft_row(
    trajectory: dict,
    split: str,
    prompt_cfg,
    prompt_template_id: str,
    prompt_view: dict | None = None,
) -> dict:
    """Render a prompt-specific SFT row from a prompt-agnostic trajectory."""
    rows = materialize_sft_rows(trajectory, split, prompt_cfg, prompt_template_id, prompt_view=prompt_view)
    if len(rows) != 1:
        raise ValueError("materialize_sft_row requires prompt_view.history_mode=full")
    return rows[0]


def materialize_sft_rows(
    trajectory: dict,
    split: str,
    prompt_cfg,
    prompt_template_id: str,
    prompt_view: dict | None = None,
) -> list[dict]:
    """Render prompt-specific SFT rows from a prompt-agnostic trajectory."""
    prompt_view = _normalize_prompt_view(prompt_view)
    system_template = prompt_cfg.get("system_template", prompt_cfg.get("system_prompt"))
    system_message = build_system_message(trajectory["container_state"], {"system_template": system_template})
    if prompt_view["history_mode"] == "full":
        return [
            _build_sft_row(
                trajectory,
                split,
                prompt_cfg,
                prompt_template_id,
                prompt_view,
                trajectory["steps"],
                system_message,
                row_step_index=None,
            )
        ]
    return [
        _build_sft_row(
            trajectory,
            split,
            prompt_cfg,
            prompt_template_id,
            prompt_view,
            trajectory["steps"][: step_idx + 1],
            system_message,
            row_step_index=step_idx,
        )
        for step_idx in range(len(trajectory["steps"]))
    ]


def _build_sft_row(
    trajectory: dict,
    split: str,
    prompt_cfg,
    prompt_template_id: str,
    prompt_view: dict,
    steps: list[dict],
    system_message: dict,
    row_step_index: int | None,
) -> dict:
    messages = [dict(system_message)]
    image_paths = []
    image_rel_paths = []
    for step in steps:
        user_content = render_user_turn_from_state(
            step["state"],
            prompt_cfg,
            include_image=prompt_view["height_map_mode"] != "none",
        )
        messages.append({"role": "user", "content": user_content})
        if prompt_view["height_map_mode"] != "none":
            image_path, image_rel_path = _materialized_image_path(step, prompt_view)
            image_paths.append({"image": image_path})
            image_rel_paths.append(image_rel_path)
        assistant_message = {
            "role": "assistant",
            "content": json.dumps(step["assistant_action"], ensure_ascii=False, separators=(",", ":")),
        }
        messages.append(assistant_message)
    messages = _apply_history_view(messages, prompt_view["history_mode"])
    visible_image_count = sum(
        1
        for message in messages
        if message.get("role") == "user" and "<image>" in str(message.get("content", ""))
    )
    if prompt_view["height_map_mode"] == "none":
        image_paths = []
        image_rel_paths = []
    elif prompt_view["history_mode"] != "full":
        image_paths = image_paths[-visible_image_count:]
        image_rel_paths = image_rel_paths[-visible_image_count:]

    return {
        "data_source": "packing_sft",
        "prompt_template_id": prompt_template_id,
        "enable_thinking": False,
        "messages": messages,
        "images": image_paths,
        "extra_info": {
            "split": split,
            "episode_id": trajectory["episode_id"],
            "difficulty": trajectory["difficulty"],
            "buffer_size": trajectory["buffer_size"],
            "container_bucket": trajectory["container_bucket"],
            "generator": trajectory["generator"],
            "num_steps": len(trajectory["steps"]),
            "row_step_index": row_step_index,
            "gt_plan": trajectory["gt_plan"],
            "final_score": trajectory["final_score"],
            "termination_reason": trajectory["termination_reason"],
            "valid_steps": trajectory["valid_steps"],
            "image_paths": image_rel_paths,
            "canonical_schema_version": trajectory["schema_version"],
            "prompt_view": prompt_view,
        },
        "canonical_trajectory": trajectory,
    }


def _normalize_prompt_view(prompt_view: dict | None) -> dict:
    prompt_view = prompt_view or {}
    history_mode = str(prompt_view.get("history_mode", "full"))
    height_map_mode = str(prompt_view.get("height_map_mode", "image"))
    valid_history_modes = {"full", "current_only", "no_assistant_history", "assistant_only_history"}
    valid_height_map_modes = {"image", "none", "blank"}
    if history_mode not in valid_history_modes:
        raise ValueError(f"invalid prompt_view.history_mode: {history_mode}")
    if height_map_mode not in valid_height_map_modes:
        raise ValueError(f"invalid prompt_view.height_map_mode: {height_map_mode}")
    return {"history_mode": history_mode, "height_map_mode": height_map_mode}


def _apply_history_view(messages: list[dict], history_mode: str) -> list[dict]:
    if history_mode == "full":
        return messages
    system = [messages[0]]
    if history_mode == "current_only":
        last_pair = messages[-2:]
        return system + last_pair
    if history_mode == "no_assistant_history":
        return [message for idx, message in enumerate(messages) if idx == len(messages) - 1 or message.get("role") != "assistant"]
    if history_mode == "assistant_only_history":
        return [
            message
            for idx, message in enumerate(messages)
            if idx == 0 or idx >= len(messages) - 2 or message.get("role") == "assistant"
        ]
    raise ValueError(f"invalid prompt_view.history_mode: {history_mode}")


def _materialized_image_path(step: dict, prompt_view: dict) -> tuple[str, str]:
    if prompt_view["height_map_mode"] == "image":
        return step["image_path"], step["image_rel_path"]
    image_path = Path(step["image_path"])
    blank_path = image_path.parent / "blank_height_map.png"
    if not blank_path.exists():
        Image.fromarray(BLANK_HEIGHT_MAP_RGB).save(blank_path)
    rel_path = str(Path(step["image_rel_path"]).parent / "blank_height_map.png")
    return str(blank_path.resolve()), rel_path


def _image_rel_dir(image_dir: Path) -> Path:
    parts = list(Path(image_dir).parts)
    if "images" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index("images")
        return Path(*parts[idx:])
    return Path("images") / Path(image_dir).name


def _without_canonical(row: dict) -> dict:
    copied = dict(row)
    copied.pop("canonical_trajectory", None)
    return copied


def _unique_canonical_trajectories(rows: list[dict]) -> list[dict]:
    seen = set()
    trajectories = []
    for row in rows:
        trajectory = row.get("canonical_trajectory")
        if not trajectory:
            continue
        episode_id = trajectory["episode_id"]
        if episode_id in seen:
            continue
        seen.add(episode_id)
        trajectories.append(trajectory)
    return trajectories


def _find_visible_object_id(visible: list[dict], internal_id: str) -> int:
    for obj in visible:
        if obj.get("internal_id") == internal_id:
            return int(obj["object_id"])
    raise RuntimeError(f"GT object {internal_id} is not visible in buffer")


def _state_from_obs(episode: dict, env: PackingEnv, visible: list[dict]) -> dict:
    buffer_objects = []
    for obj in visible:
        size_cm = [round(float(s) * 100.0, 4) for s in obj["size_m"]]
        size_cells = [
            int(round(size_cm[0] / env.xy_cell_cm)),
            int(round(size_cm[1] / env.xy_cell_cm)),
            int(round(size_cm[2] / env.z_cell_cm)),
        ]
        buffer_objects.append({"object_id": int(obj["object_id"]), "size_cm": size_cm, "size_cells": size_cells})
    return {
        "container_size_cm": episode["container_size_cm"],
        "container_grid_size": episode["container_grid_size"],
        "grid_cell_size_cm": episode["grid_cell_size_cm"],
        "buffer_objects": buffer_objects,
    }


def _draw_gt_overlay(height_rgb, box: dict, container_size: tuple[int, int, int]):
    image = Image.fromarray(height_rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    x_cells, y_cells, _ = container_size
    sx = img_w / float(x_cells)
    sy = img_h / float(y_cells)
    x0 = int(round(box["x"] * sx))
    y0 = int(round(box["y"] * sy))
    x1 = int(round((box["x"] + box["dx"]) * sx))
    y1 = int(round((box["y"] + box["dy"]) * sy))
    draw.rectangle([x0, y0, x1, y1], outline=(255, 255, 0), width=3)
    return image


def _sequence_check_report(rows: list[dict], root_dir: Path, replay_report: dict) -> dict:
    report = {
        "rows": len(rows),
        "errors": [],
        "replay_report": replay_report,
        "checked_images": 0,
    }
    for row_idx, row in enumerate(rows):
        messages = row.get("messages", [])
        images = row.get("images", [])
        if not messages or messages[0].get("role") != "system":
            report["errors"].append(f"row {row_idx}: first message is not system")
        user_turns = [msg for msg in messages if msg.get("role") == "user"]
        assistant_turns = [msg for msg in messages if msg.get("role") == "assistant"]
        if len(user_turns) != len(images):
            report["errors"].append(f"row {row_idx}: user/image count mismatch")
        if len(user_turns) != len(assistant_turns):
            report["errors"].append(f"row {row_idx}: user/assistant count mismatch")
        image_turn_count = 0
        for turn_idx, user_msg in enumerate(user_turns):
            image_refs = str(user_msg.get("content", "")).count("<image>")
            if image_refs > 1:
                report["errors"].append(f"row {row_idx} turn {turn_idx}: multiple image placeholders")
            image_turn_count += image_refs
        if image_turn_count != len(images):
            report["errors"].append(f"row {row_idx}: image placeholder/image count mismatch")
        for img in images:
            image_ref = img.get("image") if isinstance(img, dict) else img
            image_path = Path(image_ref)
            if not image_path.is_absolute():
                image_path = root_dir / image_path
            if not image_path.exists():
                report["errors"].append(f"row {row_idx}: missing image {image_path}")
                continue
            with Image.open(image_path) as image:
                if image.size != (224, 224):
                    report["errors"].append(f"row {row_idx}: unexpected image size {image.size} for {image_path}")
            report["checked_images"] += 1
        for turn_idx, assistant_msg in enumerate(assistant_turns):
            try:
                action = json.loads(assistant_msg.get("content", ""))
            except json.JSONDecodeError as exc:
                report["errors"].append(f"row {row_idx} assistant {turn_idx}: invalid json {exc}")
                continue
            if set(action) != {"object_id", "x", "y", "rotation"}:
                report["errors"].append(f"row {row_idx} assistant {turn_idx}: unexpected keys {sorted(action)}")
    report["passed"] = not report["errors"] and replay_report.get("failed") == 0 and replay_report.get("score_non_1") == 0
    return report


def _write_sequence_summary(out_dir: Path, expected_num_shards: int) -> dict:
    out_dir = Path(out_dir)
    manifests = []
    for path in sorted(out_dir.glob("**/shard_*_of_*/manifest.json")):
        manifests.append(json.loads(path.read_text(encoding="utf-8")))
    difficulty_counts = {}
    buffer_counts = {}
    failed = 0
    score_non_1 = 0
    episodes = 0
    rows = 0
    shard_keys = []
    for manifest in manifests:
        episodes += int(manifest["num_episodes"])
        rows += int(manifest.get("sequence_rows", 0))
        active_difficulties = list(manifest.get("difficulty_counts", {}).keys())
        if active_difficulties:
            shard_keys.append((active_difficulties[0], int(manifest["shard_index"])))
        for key, value in manifest.get("difficulty_counts", {}).items():
            difficulty_counts[key] = difficulty_counts.get(key, 0) + int(value)
        for key, value in manifest.get("buffer_size_counts", {}).items():
            buffer_counts[key] = buffer_counts.get(key, 0) + int(value)
        replay_report = manifest.get("replay_report", {})
        failed += int(replay_report.get("failed", 0))
        score_non_1 += int(replay_report.get("score_non_1", 0))
    difficulties_found = sorted({difficulty for difficulty, _ in shard_keys})
    expected_pairs = {
        (difficulty, shard)
        for difficulty in difficulties_found
        for shard in range(int(expected_num_shards))
    }
    found_pairs = set(shard_keys)
    summary = {
        "mode": "sft",
        "dataset_dir": str(out_dir),
        "shards_found": len(found_pairs),
        "expected_num_shards": int(expected_num_shards),
        "difficulties_found": difficulties_found,
        "shards_complete": bool(found_pairs) and found_pairs == expected_pairs,
        "episodes": episodes,
        "sequence_rows": rows,
        "failed": failed,
        "score_non_1": score_non_1,
        "difficulty_counts": difficulty_counts,
        "buffer_size_counts": buffer_counts,
    }
    write_json(summary, out_dir / "metadata_summary.json")
    return summary


def _load_env_cfg(path: str):
    path = Path(path)
    config_dir = str(path.parent.resolve())
    config_name = path.stem
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name)


def _load_cfg(path: str):
    return OmegaConf.load(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--start-shard", type=int, default=None)
    parser.add_argument("--end-shard", type=int, default=None)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--single-shard", action="store_true")
    parser.add_argument("--no-summarize-sft", action="store_true")
    parser.add_argument("--summarize-sft", action="store_true")
    parser.add_argument("--summarize-sft-sequence", action="store_true")
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.merge(_load_cfg(args.config), OmegaConf.from_dotlist(overrides))
    if args.shard_index is not None:
        cfg.shard_index = args.shard_index
    if args.num_shards is not None:
        cfg.num_shards = args.num_shards
    env_cfg = _load_env_cfg(cfg.env_config_path)
    if cfg.mode == "eval":
        manifest = build_eval_dataset(cfg)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    elif cfg.mode == "sft":
        if args.single_shard or args.shard_index is not None:
            manifest = build_sft_dataset(cfg, env_cfg)
            if args.summarize_sft or args.summarize_sft:
                _write_sequence_summary(Path(cfg.output_dir), expected_num_shards=int(cfg.get("num_shards", 1)))
            print(json.dumps(manifest, ensure_ascii=False, indent=2))
        else:
            manifests = build_sft_shards(
                cfg,
                env_cfg,
                start_shard=args.start_shard,
                end_shard=args.end_shard,
                num_shards=args.num_shards,
                skip_completed=args.skip_completed,
                summarize=not args.no_summarize_sft,
            )
            print(json.dumps({"mode": "sft", "shards_built": len(manifests)}, ensure_ascii=False, indent=2))
    elif cfg.mode == "sft_val":
        manifest = build_sft_val_dataset(cfg, env_cfg)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
    else:
        raise ValueError(f"invalid mode: {cfg.mode}")


if __name__ == "__main__":
    main()
