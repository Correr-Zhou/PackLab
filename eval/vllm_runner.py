from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import glob
import hashlib
import inspect
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import ListConfig, OmegaConf

from eval.recorder import EpisodeRecorder
from eval.report import write_eval_outputs
from phy_env.components.policy import PolicyLLMJson
from phy_env.components.scoring import PackingScore
from phy_env.env import PackingEnv
from phy_env.metrics import final_packing_metrics, object_volume
from runtime.episode_control import EpisodeControl, EpisodeControlConfig
from runtime.history import append_action_turn, append_observation_turn, init_history
from runtime.messages import build_system_message, build_user_text
from runtime.state import build_buffer_state, build_container_state

FALLBACK_INVALID_ACTION = {"object_id": -1, "x": 0, "y": 0, "rotation": 0}


def retry_request(fn, max_retries: int, retry_sleep_s: float):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(retry_sleep_s)
    raise last_exc


def evaluate_cases_with_client(cases: list[dict], client_fn, max_retries: int, retry_sleep_s: float, fail_fast: bool):
    results = []
    for case in cases:
        started = time.perf_counter()
        try:
            result = retry_request(lambda: client_fn(case), max_retries=max_retries, retry_sleep_s=retry_sleep_s)
            merged = dict(case)
            merged.update(result)
            merged["elapsed_s"] = time.perf_counter() - started
            results.append(merged)
        except Exception as exc:
            if fail_fast:
                raise
            failed = dict(case)
            failed.update(
                {
                    "status": "api_error",
                    "error": str(exc),
                    "retry_count": max_retries,
                    "elapsed_s": time.perf_counter() - started,
                }
            )
            results.append(failed)
    return results


def evaluate_cases_with_client_concurrent(
    cases: list[dict],
    client_fn,
    max_retries: int,
    retry_sleep_s: float,
    fail_fast: bool,
    concurrency: int,
):
    if concurrency <= 1:
        return evaluate_cases_with_client(cases, client_fn, max_retries, retry_sleep_s, fail_fast)

    def run_one(case):
        started = time.perf_counter()
        try:
            result = retry_request(lambda: client_fn(case), max_retries=max_retries, retry_sleep_s=retry_sleep_s)
            return result, time.perf_counter() - started, None
        except Exception as exc:
            return None, time.perf_counter() - started, exc

    results = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=int(concurrency)) as executor:
        future_to_idx = {executor.submit(run_one, case): idx for idx, case in enumerate(cases)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            case = cases[idx]
            result, elapsed_s, exc = future.result()
            if exc is None:
                merged = dict(case)
                merged.update(result)
                merged["elapsed_s"] = elapsed_s
                results[idx] = merged
            else:
                if fail_fast:
                    raise exc
                failed = dict(case)
                failed.update(
                    {"status": "api_error", "error": str(exc), "retry_count": max_retries, "elapsed_s": elapsed_s}
                )
                results[idx] = failed
    return results


def _visualization_mode(cfg) -> str:
    return str(OmegaConf.select(cfg, "visualization.mode", default="none"))


def _visualization_case_indices(cases: list[dict], cfg) -> set[int]:
    mode = _visualization_mode(cfg)
    if mode in ("none", "false", "off", "0"):
        return set()
    if mode == "all":
        return set(range(len(cases)))
    if mode != "first_n_per_difficulty":
        raise ValueError(f"invalid visualization mode: {mode}")
    limit = int(OmegaConf.select(cfg, "visualization.first_n_per_difficulty", default=0))
    if limit <= 0:
        return set()
    counts = {}
    selected = set()
    for idx, case in enumerate(cases):
        difficulty = str(case.get("difficulty", "unknown"))
        count = counts.get(difficulty, 0)
        if count < limit:
            selected.add(idx)
        counts[difficulty] = count + 1
    return selected


def _as_builtin(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _as_builtin(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_as_builtin(v) for v in value]
    return value


def load_eval_cases(
    path: str | Path | list[str],
    max_cases: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    max_cases_per_difficulty: int | None = None,
):
    paths = _expand_eval_paths(path)
    dataframe = pd.concat([pd.read_parquet(item) for item in paths], ignore_index=True)
    if shuffle:
        dataframe = dataframe.sample(frac=1.0, random_state=seed)
    if max_cases_per_difficulty is not None:
        dataframe = (
            dataframe.assign(
                _difficulty=dataframe.apply(
                    lambda row: _as_builtin(row.get("extra_info", {})).get(
                        "difficulty", row.get("difficulty", "unknown")
                    ),
                    axis=1,
                )
            )
            .groupby("_difficulty", sort=False, group_keys=False)
            .head(int(max_cases_per_difficulty))
            .drop(columns=["_difficulty"])
        )
    elif max_cases is not None:
        dataframe = dataframe.head(int(max_cases))
    cases = []
    for row_id, row in enumerate(dataframe.to_dict(orient="records")):
        extra_info = _as_builtin(row.get("extra_info", {}))
        cases.append(
            {
                "sample_id": str(extra_info.get("episode_id", extra_info.get("index", row_id))),
                "difficulty": extra_info.get("difficulty", row.get("difficulty", "unknown")),
                "buffer_size": extra_info.get("interaction_kwargs", {}).get("buffer_size"),
                "extra_info": extra_info,
            }
        )
    return cases


def _expand_eval_paths(path: str | Path | list[str]) -> list[str]:
    raw_paths = list(path) if isinstance(path, list | tuple | ListConfig) else [path]
    paths = []
    for item in raw_paths:
        pattern = str(item)
        matched = sorted(glob.glob(pattern, recursive=True))
        paths.extend(matched or [pattern])
    if not paths:
        raise FileNotFoundError(f"no eval files match: {path}")
    return paths


def _episode_control_cfg(cfg) -> EpisodeControlConfig:
    raw = cfg.get("episode_control", {})
    max_invalid_actions = raw.get("max_invalid_actions", 3)
    return EpisodeControlConfig(
        max_invalid_actions=None if max_invalid_actions is None else int(max_invalid_actions),
        max_turns_buffer=int(raw.get("max_turns_buffer", 3)),
        hard_max_turns=int(raw.get("hard_max_turns", 40)),
    )


def _load_env_cfg(path: str | Path):
    path = Path(path)
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(path.parent.resolve())):
        return compose(config_name=path.stem)


def resolve_scoring_config_path(cfg) -> str:
    return str(OmegaConf.select(cfg, "scoring.config_path", default="configs/phy_env/scoring/packing_score.yaml"))


def _format_event(parse_info: dict) -> str:
    if not parse_info.get("parse_ok", False):
        return "parse_fail"
    return "strict_json" if parse_info.get("strict_json", False) else "extra_text"


def _xy_bounds_adjustment_cfg(cfg) -> dict:
    raw = cfg.get("action_adjustment", {}).get("xy_bounds", {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "mode": str(raw.get("mode", "random")),
        "seed": int(raw.get("seed", 0)),
    }


def _height_adjustment_cfg(cfg) -> dict:
    raw = cfg.get("action_adjustment", {}).get("height", {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "mode": str(raw.get("mode", "random_feasible")),
        "max_attempts": int(raw.get("max_attempts", 32)),
        "seed": int(raw.get("seed", 0)),
    }


def _placement_mode(cfg) -> str:
    return str(OmegaConf.select(cfg, "placement.mode", default="strict"))


def _forced_outside_seed(cfg) -> int:
    return int(OmegaConf.select(cfg, "placement.forced_outside_seed", default=0))


def _policy_cfg(env_cfg, cfg):
    policy_cfg = OmegaConf.create(OmegaConf.to_container(env_cfg.policy, resolve=True))
    parsing_cfg = cfg.get("parsing", {})
    if "strip_think_tags" in parsing_cfg:
        policy_cfg.strip_think_tags = bool(parsing_cfg.strip_think_tags)
    return policy_cfg


def _reask_on_parse_fail_cfg(cfg) -> dict:
    raw = cfg.get("parsing", {}).get("reask_on_parse_fail", {})
    return {
        "enabled": bool(raw.get("enabled", False)),
        "disable_thinking": bool(raw.get("disable_thinking", False)),
        "prompt": str(
            raw.get(
                "prompt",
                'Return only one JSON object for the same current step. '
                'Do not include reasoning, markdown, code fences, or any extra text. '
                'Use exactly this schema: {"object_id": int, "x": int, "y": int, "rotation": 0 or 1}',
            )
        ),
    }


def _diagnostics_cfg(cfg) -> dict:
    raw = cfg.get("diagnostics", {})
    prompt_view = cfg.get("prompt_view", {})
    image_mode = str(prompt_view.get("height_map_mode", raw.get("image_mode", "normal")))
    history_mode = str(prompt_view.get("history_mode", raw.get("history_mode", "full")))
    if image_mode == "image":
        image_mode = "normal"
    if history_mode == "no_assistant_history":
        history_mode = "no_assistant"
    if image_mode == "none":
        image_mode = "none"
    valid_image_modes = {"normal", "blank", "solid_red", "solid_blue", "shuffled_across_cases", "stale_first_frame"}
    valid_image_modes.add("none")
    valid_history_modes = {"full", "current_only", "no_assistant", "shuffled_assistant_actions"}
    if image_mode not in valid_image_modes:
        raise ValueError(f"invalid diagnostics.image_mode: {image_mode}")
    if history_mode not in valid_history_modes:
        raise ValueError(f"invalid diagnostics.history_mode: {history_mode}")
    return {
        "log_prompt_payload": bool(raw.get("log_prompt_payload", False)),
        "image_mode": image_mode,
        "history_mode": history_mode,
        "seed": int(raw.get("seed", 0)),
    }


def _clone_messages(messages: list[dict]) -> list[dict]:
    return copy.deepcopy(messages)


def _diagnostic_image(image, diagnostics_cfg: dict, case: dict, step_index: int, first_image=None):
    mode = diagnostics_cfg["image_mode"]
    if mode == "normal":
        return image
    if mode == "none":
        return None
    if mode == "blank":
        return np.zeros_like(np.asarray(image, dtype=np.uint8))
    if mode in {"solid_red", "solid_blue"}:
        arr = np.zeros_like(np.asarray(image, dtype=np.uint8))
        if arr.ndim >= 3 and arr.shape[-1] >= 3:
            arr[..., 0 if mode == "solid_red" else 2] = 255
        return arr
    if mode == "stale_first_frame":
        return first_image if first_image is not None else image
    if mode == "shuffled_across_cases":
        arr = np.asarray(image, dtype=np.uint8)
        seed_material = f"{diagnostics_cfg.get('seed', 0)}:{case.get('sample_id')}:{step_index}:image_shuffle"
        seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        flat = arr.reshape(-1, arr.shape[-1]).copy()
        rng.shuffle(flat, axis=0)
        return flat.reshape(arr.shape)
    raise ValueError(f"invalid diagnostics.image_mode: {mode}")


def _request_history(history: list[dict], diagnostics_cfg: dict, case: dict, step_index: int) -> list[dict]:
    mode = diagnostics_cfg["history_mode"]
    messages = _clone_messages(history)
    if mode == "full":
        return messages
    system = [message for message in messages if message.get("role") == "system"]
    if mode == "current_only":
        latest_user = next((message for message in reversed(messages) if message.get("role") == "user"), None)
        return system + ([latest_user] if latest_user is not None else [])
    if mode == "no_assistant":
        return [message for message in messages if message.get("role") != "assistant"]
    if mode == "shuffled_assistant_actions":
        assistants = [message.get("content", "") for message in messages if message.get("role") == "assistant"]
        if len(assistants) <= 1:
            return messages
        seed_material = f"{diagnostics_cfg.get('seed', 0)}:{case.get('sample_id')}:{step_index}:assistant_shuffle"
        seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        shuffled = list(assistants)
        random.Random(seed).shuffle(shuffled)
        out = []
        assistant_idx = 0
        for message in messages:
            item = dict(message)
            if item.get("role") == "assistant":
                item["content"] = shuffled[assistant_idx]
                assistant_idx += 1
            out.append(item)
        return out
    raise ValueError(f"invalid diagnostics.history_mode: {mode}")


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _first_image_url(messages: list[dict]) -> str | None:
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                return str(item.get("image_url", {}).get("url", ""))
    return None


def _latest_user_text(messages: list[dict]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return _content_text(message.get("content", ""))
    return ""


def _hash_string(value: str | None) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _prompt_payload_summary(
    request_messages: list[dict],
    diagnostics_cfg: dict,
    buffer_state: list[dict],
    step_index: int,
) -> dict | None:
    if not diagnostics_cfg["log_prompt_payload"]:
        return None
    image_url = _first_image_url(request_messages)
    history_char_len = sum(len(_content_text(message.get("content", ""))) for message in request_messages)
    return {
        "image_mode": diagnostics_cfg["image_mode"],
        "history_mode": diagnostics_cfg["history_mode"],
        "step_index": int(step_index),
        "message_count": len(request_messages),
        "user_turn_count": sum(1 for message in request_messages if message.get("role") == "user"),
        "assistant_turn_count": sum(1 for message in request_messages if message.get("role") == "assistant"),
        "image_hash": _hash_string(image_url),
        "user_text_hash": _hash_string(_latest_user_text(request_messages)),
        "history_char_len": history_char_len,
        "current_buffer_object_ids": [int(item["object_id"]) for item in buffer_state],
        "current_buffer_size": len(buffer_state),
    }


def _call_client(client_fn, messages: list[dict], request_overrides: dict | None = None) -> str:
    request_messages = _clone_messages(messages)
    if not request_overrides:
        return str(client_fn(request_messages))
    try:
        signature = inspect.signature(client_fn)
    except (TypeError, ValueError):
        return str(client_fn(request_messages))
    if "request_overrides" not in signature.parameters:
        return str(client_fn(request_messages))
    return str(client_fn(request_messages, request_overrides=request_overrides))


def _reask_request_overrides(reask_cfg: dict) -> dict | None:
    if not reask_cfg.get("disable_thinking", False):
        return None
    return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def _request_overrides_from_prompt(prompt_cfg) -> dict | None:
    allow_thinking = OmegaConf.select(prompt_cfg, "allow_thinking")
    if allow_thinking is None:
        return None
    overrides = {"extra_body": {"chat_template_kwargs": {"enable_thinking": bool(allow_thinking)}}}
    if bool(allow_thinking) and _thinking_prefill(prompt_cfg):
        overrides["extra_body"]["add_generation_prompt"] = False
        overrides["extra_body"]["continue_final_message"] = True
    return overrides


def _thinking_prefill(prompt_cfg) -> str:
    if not bool(OmegaConf.select(prompt_cfg, "allow_thinking", default=False)):
        return ""
    return str(OmegaConf.select(prompt_cfg, "thinking_prefill", default="") or "")


def _append_thinking_prefill_message(messages: list[dict], prompt_cfg) -> list[dict]:
    request_messages = _clone_messages(messages)
    prefill = _thinking_prefill(prompt_cfg)
    if prefill:
        request_messages.append({"role": "assistant", "content": prefill})
    return request_messages


def _forced_action_for_visible(env: PackingEnv, case: dict, step_index: int, reason: str, seed: int) -> tuple[dict, int]:
    visible = env.buffer.get_visible_objects()
    if not visible:
        return dict(FALLBACK_INVALID_ACTION), seed
    seed_material = f"{seed}:{case.get('sample_id')}:{step_index}:{reason}"
    derived_seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(derived_seed)
    return {"object_id": rng.randrange(len(visible)), "x": 0, "y": 0, "rotation": 0}, derived_seed


def _count_reason(counter: dict, reason: str | None) -> None:
    if not reason:
        return
    for item in str(reason).split(","):
        if item:
            counter[item] = counter.get(item, 0) + 1


def _object_volume(obj: dict) -> float:
    return object_volume(obj)


def _inside_metrics(env: PackingEnv, total_object_volume: float, total_object_count: int) -> dict:
    return final_packing_metrics(env, total_object_volume, total_object_count)


def _final_placed_objects(env: PackingEnv) -> list[dict]:
    x_min, x_max, y_min, y_max, bottom_z, top_z = env._container_interior_bounds()
    final_objects = []
    for placed_index, obj in enumerate(env.get_placed_objects()):
        body_id = int(obj["body_id"])
        position, orientation = env.p.getBasePositionAndOrientation(body_id)
        aabb_min, aabb_max = env.p.getAABB(body_id)
        center_xyz_inside = (
            x_min <= float(position[0]) <= x_max
            and y_min <= float(position[1]) <= y_max
            and bottom_z <= float(position[2]) <= top_z
        )
        aabb_xy_inside = (
            x_min <= float(aabb_min[0])
            and float(aabb_max[0]) <= x_max
            and y_min <= float(aabb_min[1])
            and float(aabb_max[1]) <= y_max
        )
        aabb_z_inside = bottom_z <= float(aabb_min[2]) and float(aabb_max[2]) <= top_z
        final_objects.append(
            {
                "placed_index": placed_index,
                "body_id": body_id,
                "type_id": int(obj["type_id"]),
                "size_m": [float(v) for v in obj["size_m"]],
                "volume": _object_volume(obj),
                "position": [float(v) for v in position],
                "orientation": [float(v) for v in orientation],
                "rotation": int(obj["rotation"]),
                "aabb_min": [float(v) for v in aabb_min],
                "aabb_max": [float(v) for v in aabb_max],
                "center_xyz_inside": bool(center_xyz_inside),
                "aabb_xy_inside": bool(aabb_xy_inside),
                "aabb_z_inside": bool(aabb_z_inside),
                "aabb_inside_container": bool(aabb_xy_inside and aabb_z_inside),
            }
        )
    return final_objects


def _xy_candidates(env: PackingEnv, bid: int) -> list[dict]:
    obj = env.buffered_objects[int(bid)]
    size_m = obj["size_m"]
    x_cells, y_cells, _ = env.container_size
    candidates = []
    for rotation in (0, 1):
        if int(rotation) == 1:
            footprint_x = int(round(size_m[1] / env.xy_resolution))
            footprint_y = int(round(size_m[0] / env.xy_resolution))
        else:
            footprint_x = int(round(size_m[0] / env.xy_resolution))
            footprint_y = int(round(size_m[1] / env.xy_resolution))
        max_x = x_cells - footprint_x
        max_y = y_cells - footprint_y
        if max_x >= 0 and max_y >= 0:
            candidates.append({"rotation": rotation, "valid_x_range": [0, max_x], "valid_y_range": [0, max_y]})
    return candidates


def _adjust_xy_bounds(action: dict, env: PackingEnv, case: dict, step_index: int, adjustment_cfg: dict):
    if not adjustment_cfg.get("enabled", False):
        return dict(action), None, False
    if adjustment_cfg.get("mode") != "random":
        raise ValueError(f"invalid xy_bounds adjust mode: {adjustment_cfg.get('mode')}")
    bid = action.get("object_id")
    if bid is None or bid < 0 or bid >= env.buffer.num_visible():
        return dict(action), None, False
    original_rotation = int(action.get("rotation", 0))
    candidates = _xy_candidates(env, int(bid))
    current = next((c for c in candidates if int(c["rotation"]) == original_rotation), None)
    if current is None:
        max_x, max_y = -1, -1
    else:
        max_x, max_y = current["valid_x_range"][1], current["valid_y_range"][1]
    no_fit_reason = None
    if current is None:
        if not candidates:
            return dict(action), None, False
        no_fit_reason = f"rotation_{original_rotation}_has_no_xy_fit"

    if current is None:
        out_x = True
        out_y = True
    else:
        x = int(action.get("x", 0))
        y = int(action.get("y", 0))
        out_x = x < 0 or x > max_x
        out_y = y < 0 or y > max_y

    if not candidates:
        return dict(action), None, False
    if not (out_x or out_y or no_fit_reason):
        return dict(action), None, False
    seed_material = f"{adjustment_cfg.get('seed', 0)}:{case.get('sample_id')}:{step_index}:{bid}:{action.get('rotation', 0)}"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    candidate = rng.choice(candidates)
    adjusted = dict(action)
    adjusted["rotation"] = int(candidate["rotation"])
    adjusted["x"] = rng.randint(*candidate["valid_x_range"])
    adjusted["y"] = rng.randint(*candidate["valid_y_range"])
    if no_fit_reason:
        reason = no_fit_reason
    elif out_x and out_y:
        reason = "out_of_bounds_xy"
    elif out_x:
        reason = "out_of_bounds_x"
    else:
        reason = "out_of_bounds_y"
    return adjusted, {
        "type": "xy_bounds_random",
        "reason": reason,
        "raw_action": dict(action),
        "valid_x_range": candidate["valid_x_range"],
        "valid_y_range": candidate["valid_y_range"],
        "candidates": candidates,
        "seed": seed,
    }, True


def _action_exceeds_height(action: dict, env: PackingEnv) -> bool:
    bid = action.get("object_id")
    if bid is None or bid < 0 or bid >= env.buffer.num_visible():
        return False
    obj = env.buffered_objects[int(bid)]
    quat = env._rotation_quaternion(int(action.get("rotation", 0)))
    oriented = env._oriented_extents(obj["size_m"], quat)
    half_x, half_y, half_z = oriented / 2.0
    x0 = env.target_origin[0] + env.wall_width
    y0 = env.target_origin[1] + env.wall_width
    x_center = x0 + int(action["x"]) * env.xy_resolution + half_x
    y_center = y0 + int(action["y"]) * env.xy_resolution + half_y
    _, _, _, _, bottom_z, top_z = env._container_interior_bounds()
    height_map, x_axis, y_axis = env._scan_container_height_map()
    local_h = env._local_max_height(height_map, x_axis, y_axis, x_center, y_center, half_x, half_y)
    z_center = bottom_z + local_h + half_z
    return bool(z_center + half_z > top_z + 1e-9)


def _adjust_height(action: dict, env: PackingEnv, case: dict, step_index: int, adjustment_cfg: dict):
    if not adjustment_cfg.get("enabled", False):
        return dict(action), None, False
    if adjustment_cfg.get("mode") != "random_feasible":
        raise ValueError(f"invalid height adjust mode: {adjustment_cfg.get('mode')}")
    bid = action.get("object_id")
    if bid is None or bid < 0 or bid >= env.buffer.num_visible():
        return dict(action), None, False
    if not _action_exceeds_height(action, env):
        return dict(action), None, False
    candidates = _xy_candidates(env, int(bid))
    if not candidates:
        return dict(action), None, False
    seed_material = f"{adjustment_cfg.get('seed', 0)}:{case.get('sample_id')}:{step_index}:{bid}:{action.get('rotation', 0)}:height"
    seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    attempts = []
    for _ in range(int(adjustment_cfg.get("max_attempts", 32))):
        candidate = rng.choice(candidates)
        adjusted = dict(action)
        adjusted["rotation"] = int(candidate["rotation"])
        adjusted["x"] = rng.randint(*candidate["valid_x_range"])
        adjusted["y"] = rng.randint(*candidate["valid_y_range"])
        attempts.append(adjusted)
        if not _action_exceeds_height(adjusted, env):
            return adjusted, {
                "type": "height_random_feasible",
                "reason": "exceed_height",
                "raw_action": dict(action),
                "attempts": len(attempts),
                "max_attempts": int(adjustment_cfg.get("max_attempts", 32)),
                "candidates": candidates,
                "seed": seed,
            }, True
    return dict(action), {
        "type": "height_random_feasible_failed",
        "reason": "exceed_height",
        "raw_action": dict(action),
        "attempts": len(attempts),
        "max_attempts": int(adjustment_cfg.get("max_attempts", 32)),
        "candidates": candidates,
        "seed": seed,
    }, False


def rollout_case_with_client(case: dict, cfg, client_fn, visualization_dir: str | Path | None = None) -> dict:
    env_cfg = _load_env_cfg(cfg.env.config_path)
    prompt_cfg = OmegaConf.load(cfg.prompt.config_path)
    env = PackingEnv(env_cfg)
    recorder = None
    policy = PolicyLLMJson(_policy_cfg(env_cfg, cfg))
    scorer = PackingScore(OmegaConf.load(resolve_scoring_config_path(cfg)))
    interaction_kwargs = _as_builtin(case["extra_info"]["interaction_kwargs"])
    sample = {
        "container_size_cm": interaction_kwargs["container_size_cm"],
        "buffer_size": interaction_kwargs["buffer_size"],
        "object_sequence": interaction_kwargs["object_sequence"],
        "seed": interaction_kwargs.get("seed"),
    }
    history = None
    steps = []
    format_events = []
    invalid_action_count = 0
    raw_invalid_action_count = 0
    adjustment_count = 0
    parse_fail_count = 0
    invalid_object_id_count = 0
    forced_outside_count = 0
    relaxed_reason_counts = {}
    xy_adjustment_cfg = _xy_bounds_adjustment_cfg(cfg)
    height_adjustment_cfg = _height_adjustment_cfg(cfg)
    reask_cfg = _reask_on_parse_fail_cfg(cfg)
    request_overrides = _request_overrides_from_prompt(prompt_cfg)
    diagnostics_cfg = _diagnostics_cfg(cfg)
    placement_mode = _placement_mode(cfg)
    forced_seed = _forced_outside_seed(cfg)
    control = EpisodeControl(_episode_control_cfg(cfg), object_count=len(sample["object_sequence"]))
    total_object_volume = sum(
        float(obj["size_cm"][0]) * float(obj["size_cm"][1]) * float(obj["size_cm"][2]) / 1_000_000.0
        for obj in sample["object_sequence"]
    )
    try:
        obs = env.reset(sample)
        first_image = copy.deepcopy(obs.image)
        if visualization_dir is not None:
            save_video = bool(OmegaConf.select(cfg, "visualization.save_video", default=False))
            recorder = EpisodeRecorder(env, visualization_dir, save_video=save_video)
            recorder.begin()
        container_state = build_container_state(env, sample["container_size_cm"], prompt_cfg=prompt_cfg)
        history = init_history(build_system_message(container_state, prompt_cfg))
        termination_reason = None
        done = False
        while not done:
            buffer_state = build_buffer_state(
                obs.buffer_objects,
                env.xy_cell_cm,
                env.z_cell_cm,
                prompt_cfg=prompt_cfg,
                container_grid_size=env.container_size,
            )
            request_image = _diagnostic_image(
                obs.image,
                diagnostics_cfg,
                case,
                control.total_steps,
                first_image=first_image,
            )
            append_observation_turn(history, build_user_text(buffer_state, prompt_cfg), request_image, target="eval")
            request_history = _request_history(history, diagnostics_cfg, case, control.total_steps)
            prompt_payload = _prompt_payload_summary(
                request_history,
                diagnostics_cfg,
                buffer_state,
                control.total_steps,
            )
            client_history = _append_thinking_prefill_message(request_history, prompt_cfg)
            raw_output = _call_client(client_fn, client_history, request_overrides)
            if recorder is not None:
                recorder.before_step()
            action, parse_info = policy.parse_with_info(raw_output)
            if action is None and reask_cfg["enabled"]:
                reask_history = _clone_messages(request_history)
                reask_history.append({"role": "assistant", "content": raw_output})
                reask_history.append({"role": "user", "content": reask_cfg["prompt"]})
                raw_output = _call_client(client_fn, reask_history, _reask_request_overrides(reask_cfg))
                action, parse_info = policy.parse_with_info(raw_output)
            append_action_turn(history, raw_output)
            format_events.append(_format_event(parse_info))
            forced_outside = False
            forced_outside_reason = None
            forced_seed_value = None
            if action is None:
                parse_fail_count += 1
                if placement_mode == "relaxed_physical":
                    action, forced_seed_value = _forced_action_for_visible(
                        env, case, control.total_steps, "parse_fail", forced_seed
                    )
                    forced_outside = True
                    forced_outside_reason = "parse_fail"
                else:
                    action = dict(FALLBACK_INVALID_ACTION)
            elif (
                placement_mode == "relaxed_physical"
                and (
                    action.get("object_id") is None
                    or action.get("object_id") < 0
                    or action.get("object_id") >= env.buffer.num_visible()
                )
            ):
                invalid_object_id_count += 1
                action, forced_seed_value = _forced_action_for_visible(
                    env, case, control.total_steps, "invalid_object_id", forced_seed
                )
                forced_outside = True
                forced_outside_reason = "invalid_object_id"
            raw_action = dict(action)
            action_adjustments = []
            raw_would_be_valid = True
            if placement_mode != "relaxed_physical" and parse_info.get("parse_ok", False):
                adjusted, action_adjustment, adjusted_flag = _adjust_xy_bounds(
                    action, env, case, control.total_steps, xy_adjustment_cfg
                )
                if action_adjustment is not None:
                    action_adjustments.append(action_adjustment)
                    raw_would_be_valid = False
                if adjusted_flag:
                    action = adjusted
                    adjustment_count += 1
                adjusted, action_adjustment, adjusted_flag = _adjust_height(
                    action, env, case, control.total_steps, height_adjustment_cfg
                )
                if action_adjustment is not None:
                    action_adjustments.append(action_adjustment)
                    raw_would_be_valid = False
                if adjusted_flag:
                    action = adjusted
                    adjustment_count += 1
            if placement_mode == "relaxed_physical":
                step_result = env.step_relaxed(
                    action,
                    force_outside=forced_outside,
                    outside_seed=forced_seed_value,
                )
            else:
                step_result = env.step(action)
            if recorder is not None:
                recorder.after_step()
            valid = bool(step_result.info.get("valid", False))
            if forced_outside:
                valid = False
                forced_outside_count += 1
            if not valid:
                invalid_action_count += 1
                _count_reason(relaxed_reason_counts, forced_outside_reason or step_result.info.get("reason"))
            if not raw_would_be_valid or not valid:
                raw_invalid_action_count += 1
            control_result = control.update(valid=valid, buffer_exhausted=env.buffer.is_exhausted())
            obs = step_result.observation
            done = bool(step_result.done or control_result.done)
            termination_reason = control_result.termination_reason or step_result.info.get("termination_reason")
            if termination_reason and env.termination_reason != termination_reason:
                env.termination_reason = termination_reason
            steps.append(
                {
                    "raw_output": raw_output,
                    "raw_action": raw_action,
                    "action": action,
                    "action_adjustment": action_adjustments[-1] if action_adjustments else None,
                    "action_adjustments": action_adjustments,
                    "placement_mode": placement_mode,
                    "forced_outside": forced_outside,
                    "forced_outside_reason": forced_outside_reason,
                    "valid": valid,
                    "reason": step_result.info.get("reason"),
                    "parse_event": format_events[-1],
                    "diagnostics": prompt_payload,
                }
            )
        raw_out_of_bounds_x_count = int(relaxed_reason_counts.get("out_of_bounds_x", 0))
        raw_out_of_bounds_y_count = int(relaxed_reason_counts.get("out_of_bounds_y", 0))
        raw_exceed_height_count = int(relaxed_reason_counts.get("exceed_height", 0))
        stats = {
            "format_events": format_events,
            "invalid_action_count": invalid_action_count,
            "raw_invalid_action_count": raw_invalid_action_count,
            "adjustment_count": adjustment_count,
            "total_steps": control.total_steps,
            "termination_reason": termination_reason or env.termination_reason or "completed",
        }
        breakdown = scorer.compute_final_score_with_info(env, stats)
        inside_metrics = _inside_metrics(env, total_object_volume, len(sample["object_sequence"]))
        final_placed_objects = _final_placed_objects(env)
        return {
            **{k: v for k, v in case.items() if k != "extra_info"},
            "status": "ok",
            "termination_reason": stats["termination_reason"],
            "total_steps": control.total_steps,
            "invalid_action_count": invalid_action_count,
            "raw_invalid_action_count": raw_invalid_action_count,
            "adjustment_count": adjustment_count,
            "parse_fail_count": parse_fail_count,
            "invalid_object_id_count": invalid_object_id_count,
            "forced_outside_count": forced_outside_count,
            "raw_out_of_bounds_x_count": raw_out_of_bounds_x_count,
            "raw_out_of_bounds_y_count": raw_out_of_bounds_y_count,
            "raw_exceed_height_count": raw_exceed_height_count,
            "relaxed_reason_counts": relaxed_reason_counts,
            "format_events": format_events,
            "steps": steps,
            "score_breakdown": breakdown,
            "final_score": float(breakdown["final_score"]),
            "compactness_raw": float(breakdown["compactness_raw"]),
            "final_placed_objects": final_placed_objects,
            **inside_metrics,
        }
    finally:
        if recorder is not None:
            recorder.finish()
        env.close()


def openai_chat_client(base_url: str, model: str, api_key: str = "EMPTY", timeout_s: float = 120.0, generation_cfg=None):
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
    generation_cfg = generation_cfg or {}

    def call(messages: list[dict], request_overrides: dict | None = None) -> str:
        request_overrides = request_overrides or {}
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=float(generation_cfg.get("temperature", 0.0)),
            top_p=float(generation_cfg.get("top_p", 1.0)),
            max_tokens=int(generation_cfg.get("max_tokens", 256)),
            **request_overrides,
        )
        return response.choices[0].message.content or ""

    return call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", os.environ.get("PACKLAB_OPENAI_API_KEY", "EMPTY")))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))
    out_dir = Path(cfg.output.root_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    if args.dry_run:
        cases = [{"sample_id": "dry_run", "difficulty": "dry", "buffer_size": 0, "final_score": 0.0, "compactness_raw": 0.0}]
        write_eval_outputs(out_dir, cases)
        print(json.dumps({"status": "dry_run", "output_dir": str(out_dir)}, ensure_ascii=False))
        return

    cases = load_eval_cases(
        cfg.data.test_path,
        max_cases=cfg.data.get("max_cases"),
        shuffle=bool(cfg.data.get("shuffle", False)),
        seed=cfg.data.get("seed"),
        max_cases_per_difficulty=cfg.data.get("max_cases_per_difficulty"),
    )
    visualization_indices = _visualization_case_indices(cases, cfg)
    visualization_root = out_dir / "visualizations"
    visualization_root.mkdir(exist_ok=True)

    def run_case(case_index: int, case: dict) -> dict:
        visualization_dir = None
        if case_index in visualization_indices:
            sample_id = str(case.get("sample_id", case_index)).replace("/", "_")
            difficulty = str(case.get("difficulty", "unknown")).replace("/", "_")
            visualization_dir = visualization_root / f"{case_index:04d}_{difficulty}_{sample_id}"
        return rollout_case_with_client(case, cfg, client, visualization_dir=visualization_dir)

    client = openai_chat_client(
        args.base_url,
        args.model,
        api_key=args.api_key,
        timeout_s=float(cfg.get("client", {}).get("timeout_s", 120.0)),
        generation_cfg=cfg.get("generation", {}),
    )
    indexed_cases = [{"_case_index": idx, **case} for idx, case in enumerate(cases)]
    concurrency = int(cfg.get("client", {}).get("concurrency", 1))
    if visualization_indices:
        concurrency = 1
    results = evaluate_cases_with_client_concurrent(
        indexed_cases,
        lambda case: run_case(int(case["_case_index"]), {k: v for k, v in case.items() if k != "_case_index"}),
        max_retries=int(cfg.get("client", {}).get("max_retries", 2)),
        retry_sleep_s=float(cfg.get("client", {}).get("retry_sleep_s", 1.0)),
        fail_fast=bool(cfg.get("client", {}).get("fail_fast", False)),
        concurrency=concurrency,
    )
    results = [{k: v for k, v in result.items() if k != "_case_index"} for result in results]
    metrics = write_eval_outputs(out_dir, results)
    print(json.dumps({"status": "ok", "output_dir": str(out_dir), "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
