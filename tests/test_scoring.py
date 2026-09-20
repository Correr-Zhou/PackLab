"""Unit tests for compactness metrics and WeightedScore."""

import numpy as np
from omegaconf import OmegaConf

from phy_env.components.scoring import BboxCompactness, HeightNormCompactness, WeightedScore


class _FakeEnv:
    def __init__(self, placed, height_map, container_size, xy_resolution):
        self._placed = placed
        self._hm = np.asarray(height_map, dtype=float)
        self.container_size = container_size
        self.xy_resolution = xy_resolution

    def get_placed_objects(self):
        return self._placed

    def get_occupied_height(self):
        return float(self._hm.max()) if self._hm.size else 0.0

    def scan_height_map(self):
        return self._hm, float(self._hm.max())


def _env_with_one_box():
    # 56x56 height map with an 8x8 occupied area at 0.2m height.
    hm = np.zeros((56, 56), dtype=float)
    hm[0:8, 0:8] = 0.2
    placed = [{"size_m": (0.2, 0.2, 0.2)}]
    return _FakeEnv(placed, hm, (56, 56, 500), 0.025)


def _weighted(components, granularity="final", penalty=0.0):
    cfg = OmegaConf.create({
        "type": "weighted", "granularity": granularity, "infeasible_penalty": penalty,
        "components": components,
    })
    return WeightedScore(cfg)


def test_empty_container_zero():
    env = _FakeEnv([], np.zeros((56, 56)), (56, 56, 500), 0.025)
    assert BboxCompactness().score(env) == 0.0
    assert HeightNormCompactness().score(env) == 0.0


def test_both_metrics_in_range():
    env = _env_with_one_box()
    assert 0.0 <= BboxCompactness().score(env) <= 1.0
    assert 0.0 <= HeightNormCompactness().score(env) <= 1.0


def test_bbox_decoupled_from_container_size():
    # Bbox compactness is independent of container size, while height-normalized compactness is diluted.
    env_small = _env_with_one_box()
    hm_big = np.zeros((112, 112), dtype=float)
    hm_big[0:8, 0:8] = 0.2
    env_big = _FakeEnv([{"size_m": (0.2, 0.2, 0.2)}], hm_big, (112, 112, 500), 0.025)
    assert abs(BboxCompactness().score(env_small) - BboxCompactness().score(env_big)) < 1e-6
    assert HeightNormCompactness().score(env_big) < HeightNormCompactness().score(env_small)


def test_weighted_sum():
    # With equal weights, the result is the weighted sum of both metric scores.
    env = _env_with_one_box()
    b = HeightNormCompactness().score(env)
    c = BboxCompactness().score(env)
    r = _weighted([
        {"type": "bbox_compactness", "weight": 0.5},
        {"type": "height_norm_compactness", "weight": 0.5},
    ])
    assert abs(r.compute_final_score(env, "completed") - (0.5 * c + 0.5 * b)) < 1e-6


def test_default_bbox_weight_one():
    env = _env_with_one_box()
    r = _weighted([{"type": "bbox_compactness", "weight": 1.0}])
    assert abs(r.compute_final_score(env, "completed") - BboxCompactness().score(env)) < 1e-6


def test_infeasible_penalty_applied():
    env = _env_with_one_box()
    base = _weighted([{"type": "bbox_compactness", "weight": 1.0}], penalty=0.0).compute_final_score(env, "infeasible")
    pen = _weighted([{"type": "bbox_compactness", "weight": 1.0}], penalty=-0.5).compute_final_score(env, "infeasible")
    assert abs((base - 0.5) - pen) < 1e-6


def test_final_granularity_no_step_score():
    env = _env_with_one_box()
    r = _weighted([{"type": "bbox_compactness", "weight": 1.0}], granularity="final")
    assert r.compute_step_score(env, {}, True) == 0.0
