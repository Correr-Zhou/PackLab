"""PackingEnv smoke tests for reset, step, and termination in DIRECT mode."""

import os
import random

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

import phy_env.components  # noqa: F401
from phy_env.env import PackingEnv

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "phy_env"))


def _load_cfg():
    # Clear global Hydra state so multiple compose calls can run in one process.
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="env")


def _sample(num_objects=4, buffer_size=2):
    rng = random.Random(0)
    seq = [{"type_id": rng.randrange(20),
            "size_cm": [rng.uniform(12, 22), rng.uniform(12, 22), rng.uniform(12, 22)]}
           for _ in range(num_objects)]
    return {"container_size_cm": [140, 140, 100], "buffer_size": buffer_size, "object_sequence": seq, "seed": 0}


def test_reset_returns_observation():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    obs = env.reset(_sample())
    assert obs.image.shape == (224, 224, 3)
    assert len(obs.buffer_objects) == 2
    env.close()


def test_rollout_until_done():
    # Step through the episode with random valid coordinates until termination.
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    obs = env.reset(_sample(num_objects=4, buffer_size=2))
    rng = random.Random(1)
    steps = 0
    while not env.done and steps < 50:
        action = {"object_id": rng.randrange(len(obs.buffer_objects)),
                  "x": rng.randrange(20), "y": rng.randrange(20), "rotation": rng.randrange(2)}
        result = env.step(action)
        obs = result.observation
        steps += 1
    assert env.done
    assert env.termination_reason in ("completed", "infeasible")
    score = env.get_final_score()
    assert 0.0 <= score <= 1.0
    env.close()


def test_invalid_object_id_marks_failure():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset(_sample())
    result = env.step({"object_id": 99, "x": 0, "y": 0, "rotation": 0})
    assert result.info["valid"] is False
    assert result.info["reason"] == "invalid_object_id"
    env.close()


def test_closing_one_env_does_not_disconnect_another_env():
    cfg = _load_cfg()
    env_a = PackingEnv(cfg)
    env_b = PackingEnv(cfg)
    env_a.reset(_sample())
    env_b.reset(_sample())

    env_a.close()
    obs = env_b.observe()

    assert obs.image.shape == (224, 224, 3)
    env_b.close()


def test_scene_capture_keeps_container_wall_collision_bodies():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset(_sample())
    wall_body_ids = list(env.container_wall_body_ids)

    env.capture_scene_rgb(width=160, height=120)

    assert env.container_wall_body_ids == wall_body_ids
    assert len(wall_body_ids) == 4
    for body_id in wall_body_ids:
        mass = env.p.getDynamicsInfo(body_id, -1)[0]
        aabb_min, aabb_max = env.p.getAABB(body_id)
        assert mass == 0.0
        assert aabb_max[2] > aabb_min[2]
    env.close()


def test_container_wall_collision_bodies_are_fully_transparent():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset(_sample())

    for body_id in env.container_wall_body_ids:
        visual_shapes = env.p.getVisualShapeData(body_id)
        assert len(visual_shapes) == 1
        rgba = visual_shapes[0][7]
        assert rgba[:3] == (1.0, 1.0, 1.0)
        assert rgba[3] == 0.0
    assert not hasattr(env, "container_wall_visual_body_ids")
    env.close()


def test_relaxed_out_of_bounds_large_object_spawns_above_wall_top():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset({
        "container_size_cm": [40, 40, 30],
        "buffer_size": 1,
        "object_sequence": [{"type_id": 0, "size_cm": [50, 20, 10]}],
        "seed": 0,
    })
    env._simulate_after_insert = lambda active_body_id=None: None

    result = env.step_relaxed({"object_id": 0, "x": 0, "y": 0, "rotation": 0})

    _, _, _, _, _, top_z = env._container_interior_bounds()
    body_id = env.placed_objects[-1]["body_id"]
    pos, _ = env.p.getBasePositionAndOrientation(body_id)
    half_z = env.placed_objects[-1]["size_m"][2] / 2.0
    assert result.info["valid"] is False
    assert result.info["reason"] == "out_of_bounds_x"
    assert pos[2] >= top_z + half_z + 0.01 - 1e-9
    env.close()


def test_relaxed_repeated_stack_spawns_above_overlapping_aabb_top():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset({
        "container_size_cm": [40, 40, 20],
        "buffer_size": 1,
        "object_sequence": [
            {"type_id": 0, "size_cm": [30, 30, 30]},
            {"type_id": 1, "size_cm": [24, 24, 20]},
            {"type_id": 2, "size_cm": [18, 18, 10]},
        ],
        "seed": 0,
    })
    env._simulate_after_insert = lambda active_body_id=None: None

    first = env.step_relaxed({"object_id": 0, "x": 0, "y": 0, "rotation": 0})
    first_body_id = env.placed_objects[-1]["body_id"]
    _, first_aabb_max = env.p.getAABB(first_body_id)
    second_obj = env.buffered_objects[0]
    second_half_z = second_obj["size_m"][2] / 2.0

    second = env.step_relaxed({"object_id": 0, "x": 0, "y": 0, "rotation": 0})
    second_body_id = env.placed_objects[-1]["body_id"]
    second_pos, _ = env.p.getBasePositionAndOrientation(second_body_id)

    assert first.info["reason"] == "exceed_height"
    assert second.info["reason"] == "exceed_height"
    assert second_pos[2] >= first_aabb_max[2] + second_half_z + 0.01 - 1e-9
    env.close()


def test_height_map_raycast_detects_objects_above_container_top_plus_10cm():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset({
        "container_size_cm": [40, 40, 20],
        "buffer_size": 1,
        "object_sequence": [{"type_id": 0, "size_cm": [30, 30, 30]}],
        "seed": 0,
    })
    env._simulate_after_insert = lambda active_body_id=None: None

    result = env.step_relaxed({"object_id": 0, "x": 0, "y": 0, "rotation": 0})
    body_id = env.placed_objects[-1]["body_id"]
    _, aabb_max = env.p.getAABB(body_id)
    _, _, _, _, bottom_z, _ = env._container_interior_bounds()
    height_map, _ = env.scan_height_map()

    assert result.info["reason"] == "exceed_height"
    assert height_map.max() >= aabb_max[2] - bottom_z - 1e-6
    env.close()


def test_height_map_display_range_uses_fixed_two_container_heights():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset({
        "container_size_cm": [40, 40, 20],
        "buffer_size": 1,
        "object_sequence": [{"type_id": 0, "size_cm": [30, 30, 20]}],
        "seed": 0,
    })

    _, display_h_max = env.scan_height_map()
    _, _, _, _, bottom_z, top_z = env._container_interior_bounds()

    assert display_h_max == 2.0 * (top_z - bottom_z)
    env.close()


def test_forced_outside_places_object_left_of_buffer_area():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset({
        "container_size_cm": [40, 40, 30],
        "buffer_size": 1,
        "object_sequence": [
            {"type_id": 0, "size_cm": [16, 16, 16]},
            {"type_id": 1, "size_cm": [16, 16, 16]},
        ],
        "seed": 0,
    })
    env._simulate_after_insert = lambda active_body_id=None: None

    result = env.step_relaxed(
        {"object_id": 0, "x": 0, "y": 0, "rotation": 0},
        force_outside=True,
        outside_seed=0,
    )

    body_id = env.placed_objects[-1]["body_id"]
    aabb_min, aabb_max = env.p.getAABB(body_id)
    buffer_x_min, _, buffer_y_min, buffer_y_max = env.gray_buffer_bounds_xy
    x_min, _, _, _, _, _ = env._container_interior_bounds()

    assert result.info["valid"] is False
    assert result.info["reason"] == "forced_outside"
    assert aabb_max[0] < buffer_x_min
    assert aabb_max[0] < x_min
    assert buffer_y_min <= (aabb_min[1] + aabb_max[1]) / 2.0 <= buffer_y_max
    env.close()


def test_forced_outside_adds_visual_only_boundary_marker():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset({
        "container_size_cm": [40, 40, 30],
        "buffer_size": 1,
        "object_sequence": [
            {"type_id": 0, "size_cm": [16, 16, 16]},
            {"type_id": 1, "size_cm": [16, 16, 16]},
        ],
        "seed": 0,
    })
    env._simulate_after_insert = lambda active_body_id=None: None

    result = env.step_relaxed(
        {"object_id": 0, "x": 0, "y": 0, "rotation": 0},
        force_outside=True,
        outside_seed=0,
    )
    placed = env.placed_objects[-1]

    assert result.info["reason"] == "forced_outside"
    assert placed.get("forced_outside") is True
    marker_ids = placed.get("forced_outside_marker_body_ids")
    assert marker_ids
    for marker_id in marker_ids:
        assert env.p.getNumJoints(marker_id) == 0
        assert env.p.getDynamicsInfo(marker_id, -1)[0] == 0.0
        visual_shapes = env.p.getVisualShapeData(marker_id)
        assert visual_shapes
        rgba = visual_shapes[0][7]
        assert rgba[0] >= 0.9 and rgba[1] >= 0.75 and rgba[2] <= 0.1 and rgba[3] >= 0.8
        assert env.p.getCollisionShapeData(marker_id, -1) == ()
    env.close()


def _assert_outside_boundary_marker(env, placed):
    marker_ids = placed.get("outside_marker_body_ids")
    assert marker_ids
    for marker_id in marker_ids:
        assert env.p.getNumJoints(marker_id) == 0
        assert env.p.getDynamicsInfo(marker_id, -1)[0] == 0.0
        visual_shapes = env.p.getVisualShapeData(marker_id)
        assert visual_shapes
        rgba = visual_shapes[0][7]
        assert rgba[0] >= 0.9 and rgba[1] >= 0.75 and rgba[2] <= 0.1 and rgba[3] >= 0.8
        assert env.p.getCollisionShapeData(marker_id, -1) == ()


def test_relaxed_aabb_exceeds_height_but_center_xyz_inside_does_not_add_outside_boundary_marker():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset({
        "container_size_cm": [40, 40, 20],
        "buffer_size": 1,
        "object_sequence": [{"type_id": 0, "size_cm": [30, 30, 30]}],
        "seed": 0,
    })
    env._simulate_after_insert = lambda active_body_id=None: None

    result = env.step_relaxed({"object_id": 0, "x": 0, "y": 0, "rotation": 0})
    placed = env.placed_objects[-1]
    _, _, _, _, bottom_z, top_z = env._container_interior_bounds()
    pos, _ = env.p.getBasePositionAndOrientation(placed["body_id"])
    _, aabb_max = env.p.getAABB(placed["body_id"])

    assert result.info["reason"] == "exceed_height"
    assert bottom_z <= pos[2] <= top_z
    assert aabb_max[2] > top_z
    assert placed.get("outside_marker_body_ids") is None
    env.close()


def test_relaxed_center_xyz_outside_height_adds_outside_boundary_marker():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset({
        "container_size_cm": [40, 40, 20],
        "buffer_size": 1,
        "object_sequence": [
            {"type_id": 0, "size_cm": [30, 30, 30]},
            {"type_id": 1, "size_cm": [24, 24, 20]},
        ],
        "seed": 0,
    })
    env._simulate_after_insert = lambda active_body_id=None: None

    env.step_relaxed({"object_id": 0, "x": 0, "y": 0, "rotation": 0})
    result = env.step_relaxed({"object_id": 0, "x": 0, "y": 0, "rotation": 0})
    placed = env.placed_objects[-1]
    _, _, _, _, _, top_z = env._container_interior_bounds()
    pos, _ = env.p.getBasePositionAndOrientation(placed["body_id"])

    assert result.info["reason"] == "exceed_height"
    assert pos[2] > top_z
    _assert_outside_boundary_marker(env, placed)
    env.close()


def test_relaxed_model_out_of_bounds_adds_outside_boundary_marker():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset({
        "container_size_cm": [40, 40, 30],
        "buffer_size": 1,
        "object_sequence": [{"type_id": 0, "size_cm": [16, 16, 16]}],
        "seed": 0,
    })
    env._simulate_after_insert = lambda active_body_id=None: None

    result = env.step_relaxed({"object_id": 0, "x": 20, "y": 0, "rotation": 0})
    placed = env.placed_objects[-1]

    assert result.info["reason"] == "out_of_bounds_x"
    _assert_outside_boundary_marker(env, placed)
    env.close()


def test_relaxed_valid_center_xyz_placement_does_not_add_outside_boundary_marker():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset({
        "container_size_cm": [40, 40, 30],
        "buffer_size": 1,
        "object_sequence": [{"type_id": 0, "size_cm": [16, 16, 16]}],
        "seed": 0,
    })

    result = env.step_relaxed({"object_id": 0, "x": 0, "y": 0, "rotation": 0})
    placed = env.placed_objects[-1]

    assert result.info["valid"] is True
    assert result.info["reason"] is None
    assert placed.get("outside_marker_body_ids") is None
    env.close()


def test_repeated_forced_outside_objects_tile_without_xy_overlap():
    cfg = _load_cfg()
    env = PackingEnv(cfg)
    env.reset({
        "container_size_cm": [40, 40, 30],
        "buffer_size": 1,
        "object_sequence": [{"type_id": i, "size_cm": [16, 16, 16]} for i in range(9)],
        "seed": 0,
    })
    env._simulate_after_insert = lambda active_body_id=None: None

    aabbs = []
    for seed in range(9):
        result = env.step_relaxed(
            {"object_id": 0, "x": 0, "y": 0, "rotation": 0},
            force_outside=True,
            outside_seed=seed,
        )
        assert result.info["reason"] == "forced_outside"
        body_id = env.placed_objects[-1]["body_id"]
        aabb_min, aabb_max = env.p.getAABB(body_id)
        aabbs.append((aabb_min, aabb_max))

    for i, (a_min, a_max) in enumerate(aabbs):
        for b_min, b_max in aabbs[i + 1:]:
            overlap_xy = not (
                a_max[0] <= b_min[0]
                or a_min[0] >= b_max[0]
                or a_max[1] <= b_min[1]
                or a_min[1] >= b_max[1]
            )
            assert not overlap_xy
    env.close()
