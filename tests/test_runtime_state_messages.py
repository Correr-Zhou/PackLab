from omegaconf import OmegaConf

from runtime.messages import build_assistant_message, build_system_message, build_user_text
from runtime.state import build_buffer_state, build_container_state


class _FakeEnv:
    container_size = (40, 32, 350)
    xy_cell_cm = 2.5
    z_cell_cm = 0.2


def test_container_state_contains_only_container_and_grid_fields():
    state = build_container_state(_FakeEnv(), container_size_cm=[100.0, 80.0, 70.0])
    assert state == {
        "container_size_cm": [100.0, 80.0, 70.0],
        "container_grid_size": {"x_cells": 40, "y_cells": 32, "z_cells": 350},
        "grid_cell_size_cm": {"xy": 2.5, "z": 0.2},
    }


def test_buffer_state_hides_internal_fields():
    obs_objects = [
        {
            "object_id": 1,
            "buffer_id": 1,
            "internal_id": "obj_00001",
            "gt_index": 1,
            "type_id": 7,
            "size_m": (0.2, 0.3, 0.12),
        }
    ]
    state = build_buffer_state(obs_objects, xy_cell_cm=2.5, z_cell_cm=0.2)
    assert state == [{"object_id": 1, "size_cm": [20.0, 30.0, 12.0], "size_cells": [8, 12, 60]}]
    assert "internal_id" not in str(state)
    assert "type_id" not in str(state)
    assert "gt_index" not in str(state)


def test_grid_only_prompt_state_omits_cm_fields():
    cfg = OmegaConf.load("configs/prompts/packing_sft.yaml")
    obs_objects = [
        {
            "object_id": 1,
            "buffer_id": 1,
            "internal_id": "obj_00001",
            "gt_index": 1,
            "type_id": 7,
            "size_m": (0.2, 0.3, 0.12),
        }
    ]

    container_state = build_container_state(_FakeEnv(), container_size_cm=[100.0, 80.0, 70.0], prompt_cfg=cfg)
    buffer_state = build_buffer_state(obs_objects, xy_cell_cm=2.5, z_cell_cm=0.2, prompt_cfg=cfg)
    system = build_system_message(container_state, cfg)
    user_text = build_user_text(buffer_state, cfg)

    assert container_state == {"container_grid_size": {"x_cells": 40, "y_cells": 32, "z_cells": 350}}
    assert buffer_state == [{"object_id": 1, "size_cells": [8, 12, 60]}]
    assert "container_size_cm" not in system["content"]
    assert "grid_cell_size_cm" not in system["content"]
    assert "size_cm" not in user_text
    assert "size_cells" in user_text


def test_grid_only_prompt_state_includes_valid_ranges_when_requested():
    cfg = OmegaConf.create(
        {
            "state_fields": {
                "container": {"include_container_grid_size": True},
                "buffer": {"include_size_cells": True, "include_valid_ranges": True},
            }
        }
    )
    obs_objects = [
        {
            "object_id": 1,
            "buffer_id": 1,
            "internal_id": "obj_00001",
            "gt_index": 1,
            "type_id": 7,
            "size_m": (0.2, 0.3, 0.12),
        }
    ]

    buffer_state = build_buffer_state(
        obs_objects,
        xy_cell_cm=2.5,
        z_cell_cm=0.2,
        prompt_cfg=cfg,
        container_grid_size={"x_cells": 40, "y_cells": 32, "z_cells": 350},
    )

    assert buffer_state == [
        {
            "object_id": 1,
            "size_cm": [20.0, 30.0, 12.0],
            "size_cells": [8, 12, 60],
            "valid_ranges": {
                "rotation_0": {"x": [0, 32], "y": [0, 20]},
                "rotation_1": {"x": [0, 28], "y": [0, 24]},
            },
        }
    ]


def test_runtime_state_handles_flat_training_prompt_fields():
    cfg = OmegaConf.create(
        {
            "state_fields": {
                "include_container_size_cm": False,
                "include_container_grid_size": True,
                "include_grid_cell_size_cm": False,
                "include_buffer_object_size_cm": False,
                "include_buffer_object_size_cells": True,
            }
        }
    )
    obs_objects = [
        {
            "object_id": 1,
            "buffer_id": 1,
            "internal_id": "obj_00001",
            "gt_index": 1,
            "type_id": 7,
            "size_m": (0.2, 0.3, 0.12),
        }
    ]

    container_state = build_container_state(_FakeEnv(), container_size_cm=[100.0, 80.0, 70.0], prompt_cfg=cfg)
    buffer_state = build_buffer_state(obs_objects, xy_cell_cm=2.5, z_cell_cm=0.2, prompt_cfg=cfg)

    assert container_state == {"container_grid_size": {"x_cells": 40, "y_cells": 32, "z_cells": 350}}
    assert buffer_state == [{"object_id": 1, "size_cells": [8, 12, 60]}]


def test_history_prompt_puts_container_in_system_and_only_buffer_in_user():
    cfg = OmegaConf.create(
        {
            "system_template": "Container:\n{container_json}\nReturn JSON only.",
            "user_template": "<image>\nCurrent visible buffer objects:\n{buffer_objects_json}",
        }
    )
    container_state = {
        "container_size_cm": [100.0, 80.0, 70.0],
        "container_grid_size": {"x_cells": 40, "y_cells": 32, "z_cells": 350},
        "grid_cell_size_cm": {"xy": 2.5, "z": 0.2},
    }
    buffer_state = [{"object_id": 0, "size_cm": [20.0, 30.0, 12.0], "size_cells": [8, 12, 60]}]

    system = build_system_message(container_state, cfg)
    user_text = build_user_text(buffer_state, cfg)
    assistant = build_assistant_message('{"object_id":0,"x":0,"y":0,"rotation":0}')

    assert system["role"] == "system"
    assert "container_grid_size" in system["content"]
    assert user_text.startswith("<image>")
    assert "buffer_objects" not in user_text
    assert "container_grid_size" not in user_text
    assert '"object_id":0' in assistant["content"]


def test_public_prompt_uses_system_user_contract():
    train_cfg = OmegaConf.load("configs/prompts/packing_sft.yaml")

    train_system = train_cfg.system_template
    assert "## Role and Goal" in train_system
    assert "You are a 3D bin-packing agent." in train_system
    assert "## Container" in train_system
    assert "The container is described by this JSON object:" in train_system
    assert "{container_json}" in train_system
    assert "container_grid_size" in train_system
    assert "## Step Input" in train_system
    assert "At each step, the user message contains:" in train_system
    assert "Current visible buffer objects" in train_system
    assert "## Required Output" in train_system
    assert "Return exactly one JSON object" in train_system
    assert '{"object_id": int, "x": int, "y": int, "rotation": 0 or 1}' in train_system
    assert "## Rules" in train_system

    assert train_cfg.user_template.startswith("<image>")
    assert "{buffer_objects_json}" in train_cfg.user_template
    assert "{state_json}" not in train_cfg.user_template
    assert "top-left anchor" in train_system
    assert "center-xyz inside-container volume ratio" in train_system


def test_public_prompt_requires_short_closed_thinking_then_json():
    cfg = OmegaConf.load("configs/prompts/packing_sft.yaml")
    system = cfg.system_template

    assert cfg.allow_thinking is True
    assert "The assistant turn starts inside an open <think> block" in system
    assert "at most 3 concise lines" in system
    assert "Do not re-read or restate the prompt" in system
    assert "Always close the thinking block with </think>" in system
    assert "Immediately after </think>, output exactly one JSON object" in system
