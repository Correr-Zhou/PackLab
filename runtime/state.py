from __future__ import annotations


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


def build_valid_ranges(size_cells: list[int], container_grid_size: dict | tuple | list | None) -> dict:
    if container_grid_size is None:
        return {}
    if isinstance(container_grid_size, dict):
        x_cells = int(container_grid_size["x_cells"])
        y_cells = int(container_grid_size["y_cells"])
    else:
        x_cells = int(container_grid_size[0])
        y_cells = int(container_grid_size[1])

    dx = int(size_cells[0])
    dy = int(size_cells[1])
    ranges = {}
    for rotation, footprint_x, footprint_y in ((0, dx, dy), (1, dy, dx)):
        max_x = x_cells - footprint_x
        max_y = y_cells - footprint_y
        if max_x >= 0 and max_y >= 0:
            ranges[f"rotation_{rotation}"] = {"x": [0, int(max_x)], "y": [0, int(max_y)]}
    return ranges


def build_container_state(env, container_size_cm: list[float], prompt_cfg=None) -> dict:
    x_cells, y_cells, z_cells = env.container_size
    state = {}
    if bool(
        _select_any(
            prompt_cfg,
            ["state_fields.container.include_container_size_cm", "state_fields.include_container_size_cm"],
            True,
        )
    ):
        state["container_size_cm"] = [float(v) for v in container_size_cm]
    if bool(
        _select_any(
            prompt_cfg,
            ["state_fields.container.include_container_grid_size", "state_fields.include_container_grid_size"],
            True,
        )
    ):
        state["container_grid_size"] = {"x_cells": int(x_cells), "y_cells": int(y_cells), "z_cells": int(z_cells)}
    if bool(
        _select_any(
            prompt_cfg,
            ["state_fields.container.include_grid_cell_size_cm", "state_fields.include_grid_cell_size_cm"],
            True,
        )
    ):
        state["grid_cell_size_cm"] = {"xy": float(env.xy_cell_cm), "z": float(env.z_cell_cm)}
    return state


def build_buffer_state(
    buffer_objects: list[dict],
    xy_cell_cm: float,
    z_cell_cm: float,
    prompt_cfg=None,
    container_grid_size: dict | tuple | list | None = None,
) -> list[dict]:
    visible = []
    include_size_cm = bool(
        _select_any(
            prompt_cfg,
            ["state_fields.buffer.include_size_cm", "state_fields.include_buffer_object_size_cm"],
            True,
        )
    )
    include_size_cells = bool(
        _select_any(
            prompt_cfg,
            ["state_fields.buffer.include_size_cells", "state_fields.include_buffer_object_size_cells"],
            True,
        )
    )
    include_valid_ranges = bool(
        _select_any(
            prompt_cfg,
            ["state_fields.buffer.include_valid_ranges", "state_fields.include_buffer_object_valid_ranges"],
            False,
        )
    )
    for obj in buffer_objects:
        object_id = int(obj.get("object_id", obj.get("buffer_id")))
        size_cm = [round(float(v) * 100.0, 4) for v in obj["size_m"]]
        size_cells = [
            int(round(size_cm[0] / xy_cell_cm)),
            int(round(size_cm[1] / xy_cell_cm)),
            int(round(size_cm[2] / z_cell_cm)),
        ]
        item = {"object_id": object_id}
        if include_size_cm:
            item["size_cm"] = size_cm
        if include_size_cells:
            item["size_cells"] = size_cells
        if include_valid_ranges:
            item["valid_ranges"] = build_valid_ranges(size_cells, container_grid_size)
        visible.append(item)
    return visible
