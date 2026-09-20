"""Prompt renderer tests."""

from omegaconf import OmegaConf

from data_engine.prompts.renderer import render_user_prompt, state_for_prompt


def test_renderer_uses_whitelisted_state_fields_only():
    cfg = OmegaConf.create({
        "user_template": "Current state:\n{state_json}",
        "state_fields": {
            "include_container_size_cm": True,
            "include_container_grid_size": True,
            "include_grid_cell_size_cm": True,
            "include_buffer_object_size_cm": True,
            "include_buffer_object_size_cells": True,
        },
    })
    row = {
        "state": {
            "container_size_cm": [100.0, 80.0, 70.0],
            "container_grid_size": {"x_cells": 40, "y_cells": 32, "z_cells": 350},
            "grid_cell_size_cm": {"xy": 2.5, "z": 0.2},
            "buffer_objects": [
                {
                    "object_id": 0,
                    "size_cm": [20.0, 30.0, 8.0],
                    "size_cells": [8, 12, 40],
                    "internal_id": "obj_0000",
                    "type_id": 3,
                }
            ],
        },
        "extra_info": {"gt_plan": [{"secret": True}], "difficulty": "hard"},
    }

    visible = state_for_prompt(row, cfg)
    text = render_user_prompt(row, cfg)

    assert visible["buffer_objects"] == [{"object_id": 0, "size_cm": [20.0, 30.0, 8.0], "size_cells": [8, 12, 40]}]
    assert "internal_id" not in text
    assert "type_id" not in text
    assert "gt_plan" not in text
    assert "difficulty" not in text
    assert "x_cells" in text


def test_renderer_can_include_valid_ranges_from_grid_size():
    cfg = OmegaConf.create({
        "user_template": "Current visible buffer objects:\n{buffer_objects_json}",
        "state_fields": {
            "include_buffer_object_size_cells": True,
            "include_buffer_object_valid_ranges": True,
        },
    })
    row = {
        "state": {
            "container_grid_size": {"x_cells": 40, "y_cells": 32, "z_cells": 350},
            "buffer_objects": [
                {
                    "object_id": 0,
                    "size_cm": [20.0, 30.0, 8.0],
                    "size_cells": [8, 12, 40],
                }
            ],
        }
    }

    visible = state_for_prompt(row, cfg)
    text = render_user_prompt(row, cfg)

    assert visible["buffer_objects"] == [
        {
            "object_id": 0,
            "size_cells": [8, 12, 40],
            "valid_ranges": {
                "rotation_0": {"x": [0, 32], "y": [0, 20]},
                "rotation_1": {"x": [0, 28], "y": [0, 24]},
            },
        }
    ]
    assert '"valid_ranges"' in text
    assert '"rotation_0"' in text
