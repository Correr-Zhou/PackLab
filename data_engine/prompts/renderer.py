"""Prompt rendering from structured dataset rows."""

from __future__ import annotations

import json

from omegaconf import DictConfig

from runtime.state import build_valid_ranges


def _select(cfg, path: str, default):
    if cfg is None:
        return default
    current = cfg
    for key in path.split("."):
        if not hasattr(current, "get"):
            return default
        current = current.get(key, None)
        if current is None:
            return default
    return current


def _select_any(cfg, paths: list[str], default):
    for path in paths:
        value = _select(cfg, path, None)
        if value is not None:
            return value
    return default


def state_for_prompt(row: dict, cfg: DictConfig) -> dict:
    state = row["state"]
    fields = cfg.get("state_fields", {})
    visible = {}
    if _select_any(fields, ["container.include_container_size_cm", "include_container_size_cm"], False):
        visible["container_size_cm"] = state["container_size_cm"]
    if _select_any(fields, ["container.include_container_grid_size", "include_container_grid_size"], False):
        visible["container_grid_size"] = state["container_grid_size"]
    if _select_any(fields, ["container.include_grid_cell_size_cm", "include_grid_cell_size_cm"], False):
        visible["grid_cell_size_cm"] = state["grid_cell_size_cm"]
    include_size_cm = _select_any(fields, ["buffer.include_size_cm", "include_buffer_object_size_cm"], False)
    include_size_cells = _select_any(fields, ["buffer.include_size_cells", "include_buffer_object_size_cells"], False)
    include_valid_ranges = _select_any(
        fields,
        ["buffer.include_valid_ranges", "include_buffer_object_valid_ranges"],
        False,
    )
    if (
        include_size_cm
        or include_size_cells
        or include_valid_ranges
    ):
        objects = []
        for obj in state["buffer_objects"]:
            item = {"object_id": obj["object_id"]}
            if include_size_cm:
                item["size_cm"] = obj["size_cm"]
            if include_size_cells:
                item["size_cells"] = obj["size_cells"]
            if include_valid_ranges:
                item["valid_ranges"] = build_valid_ranges(obj["size_cells"], state.get("container_grid_size"))
            objects.append(item)
        visible["buffer_objects"] = objects
    return visible


def render_user_prompt(row: dict, cfg: DictConfig) -> str:
    state = state_for_prompt(row, cfg)
    state_json = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    buffer_objects_json = json.dumps(state.get("buffer_objects", []), ensure_ascii=False, separators=(",", ":"))
    return str(cfg["user_template"]).replace("{state_json}", state_json).replace(
        "{buffer_objects_json}", buffer_objects_json
    )


def render_user_turn_from_state(state: dict, cfg: DictConfig, include_image: bool = True) -> str:
    rendered = render_user_prompt({"state": state}, cfg)
    if include_image:
        return rendered if rendered.lstrip().startswith("<image>") else "<image>\n" + rendered
    return rendered.replace("<image>", "", 1).lstrip()
