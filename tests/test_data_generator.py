"""Data generator tests for constructed layer-cuboid episodes."""

import json

from data_engine.build_dataset import _write_sequence_summary
from data_engine.generators.layer_cuboid import GenerationConfig, LayerCuboidGenerator


def test_generated_episode_is_grid_aligned_and_in_difficulty_range():
    cfg = GenerationConfig(num_episodes=1, difficulties={"medium": 1.0}, buffer_sizes=[5], seed=123)
    episode = LayerCuboidGenerator(cfg).generate_episode(0)

    assert episode["difficulty"] == "medium"
    assert 9 <= len(episode["object_sequence"]) <= 18
    assert episode["buffer_size"] == 5

    x_cm, y_cm, z_cm = episode["container_size_cm"]
    assert (x_cm / 2.5).is_integer()
    assert (y_cm / 2.5).is_integer()
    assert (z_cm / 0.2).is_integer()

    for obj in episode["object_sequence"]:
        l_cm, w_cm, h_cm = obj["size_cm"]
        assert (l_cm / 2.5).is_integer()
        assert (w_cm / 2.5).is_integer()
        assert (h_cm / 0.2).is_integer()


def test_each_layer_covers_full_container_floor_without_overlap():
    cfg = GenerationConfig(num_episodes=1, difficulties={"hard": 1.0}, buffer_sizes=[10], seed=7)
    episode = LayerCuboidGenerator(cfg).generate_episode(0)
    x_cells = episode["container_grid_size"]["x_cells"]
    y_cells = episode["container_grid_size"]["y_cells"]

    by_layer = {}
    for item in episode["gt_plan"]:
        box = item["target_box_cells"]
        by_layer.setdefault(box["z"], []).append(box)

    assert 3 <= len(by_layer) <= 4
    for boxes in by_layer.values():
        occupied = set()
        for box in boxes:
            for x in range(box["x"], box["x"] + box["dx"]):
                for y in range(box["y"], box["y"] + box["dy"]):
                    cell = (x, y)
                    assert cell not in occupied
                    occupied.add(cell)
        assert len(occupied) == x_cells * y_cells


def test_generated_gt_plan_matches_object_sequence_order():
    cfg = GenerationConfig(num_episodes=1, difficulties={"easy": 1.0}, buffer_sizes=[3], seed=99)
    episode = LayerCuboidGenerator(cfg).generate_episode(0)

    object_ids = [obj["internal_id"] for obj in episode["object_sequence"]]
    plan_ids = [step["internal_id"] for step in episode["gt_plan"]]
    assert object_ids == plan_ids
    assert all(step["action"]["object_id"] is None for step in episode["gt_plan"])


def test_difficulty_specific_buffer_and_length_mapping():
    expected = {
        "easy": (3, 4, 8),
        "medium": (5, 9, 18),
        "hard": (10, 19, 32),
    }
    for difficulty, (buffer_size, min_len, max_len) in expected.items():
        cfg = GenerationConfig(num_episodes=5, difficulties={difficulty: 1.0}, buffer_sizes=[buffer_size], seed=1234)
        episodes = LayerCuboidGenerator(cfg).generate()
        for episode in episodes:
            assert episode["difficulty"] == difficulty
            assert episode["buffer_size"] == buffer_size
            assert min_len <= len(episode["object_sequence"]) <= max_len


def test_first_anchor_distribution_is_not_all_top_left():
    cfg = GenerationConfig(
        num_episodes=200,
        difficulties={"easy": 1.0},
        buffer_sizes=[3],
        seed=2024,
        first_anchor_distribution={
            "top_left": 0.225,
            "top_right": 0.225,
            "bottom_left": 0.225,
            "bottom_right": 0.225,
            "edge_or_interior": 0.10,
        },
    )
    episodes = LayerCuboidGenerator(cfg).generate()
    first_positions = [(ep["gt_plan"][0]["action"]["x"], ep["gt_plan"][0]["action"]["y"]) for ep in episodes]

    assert len(set(first_positions)) > 3
    assert sum(1 for x, y in first_positions if x == 0 and y == 0) < 80


def test_sequence_summary_skips_empty_difficulty_manifests(tmp_path):
    shard_dir = tmp_path / "train" / "medium" / "shard_00000_of_00001"
    shard_dir.mkdir(parents=True)
    (shard_dir / "manifest.json").write_text(
        json.dumps(
            {
                "num_episodes": 0,
                "sequence_rows": 0,
                "shard_index": 0,
                "difficulty_counts": {},
                "buffer_size_counts": {},
                "replay_report": {"failed": 0, "score_non_1": 0},
            }
        ),
        encoding="utf-8",
    )

    summary = _write_sequence_summary(tmp_path, expected_num_shards=1)

    assert summary["shards_found"] == 0
    assert summary["episodes"] == 0
    assert summary["sequence_rows"] == 0
