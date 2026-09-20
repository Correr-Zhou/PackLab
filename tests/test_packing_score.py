import pytest
from omegaconf import OmegaConf

from phy_env.components.scoring import PackingScore


class _FakePhysics:
    def getAABB(self, body_id):
        return (0.0, 0.0, 0.0), (1.0, 1.0, 0.5 if body_id == 1 else 2.0)


class _FakeEnv:
    termination_reason = "completed"
    p = _FakePhysics()

    def get_placed_objects(self):
        return [
            {
                "body_id": 1,
                "position": (0.5, 0.5, 0.25),
                "size_m": (1.0, 1.0, 0.5),
            },
            {
                "body_id": 2,
                "position": (1.5, 0.5, 0.25),
                "size_m": (1.0, 1.0, 0.5),
            },
        ]

    def _container_interior_bounds(self):
        return 0.0, 1.0, 0.0, 1.0, 0.0, 1.0


def _scorer():
    cfg = OmegaConf.create(
        {
            "type": "packing_score",
            "granularity": "final",
            "quality": {
                "enable": True,
                "use_center": "xyz",
                "inside_volume_ratio_weight": 0.7,
                "inside_container_compactness_weight": 0.3,
            },
            "format": {
                "enable": True,
                "weight": 1.0,
                "aggregate": "mean",
                "strict_json_score": 0.0,
                "extra_text_score": -0.02,
                "parse_fail_score": -0.10,
            },
            "failure": {
                "enable": True,
                "xyz_out_of_bounds_score": -0.10,
            },
        }
    )
    return PackingScore(cfg)


def test_completed_strict_json_has_no_positive_format_bonus():
    stats = {
        "format_events": ["strict_json", "strict_json"],
        "xyz_out_of_bounds_count": 0,
        "total_steps": 2,
        "total_object_volume": 1.0,
        "total_object_count": 2,
        "termination_reason": "completed",
    }
    result = _scorer().compute_final_score_with_info(_FakeEnv(), stats)
    assert result["center_xyz_inside_volume_ratio"] == 0.5
    assert result["center_xyz_inside_container_compactness"] == 1.0
    assert result["center_xyz_overall_packing_score"] == 0.5
    assert result["quality_inside_volume_part"] == 0.35
    assert result["quality_compactness_part"] == 0.3
    assert result["format_part"] == 0.0
    assert result["xyz_out_of_bounds_part"] == 0.0
    assert result["final_score"] == pytest.approx(0.65)


def test_format_and_xyz_out_of_bounds_penalties_are_applied_without_terminal_cliff():
    stats = {
        "format_events": ["extra_text", "parse_fail"],
        "invalid_action_count": 2,
        "xyz_out_of_bounds_count": 1,
        "total_steps": 2,
        "total_object_volume": 1.0,
        "total_object_count": 2,
        "termination_reason": "max_invalid_actions",
    }
    result = _scorer().compute_final_score_with_info(_FakeEnv(), stats)
    assert result["format_part"] == pytest.approx(-0.06)
    assert result["xyz_out_of_bounds_count"] == 1
    assert result["xyz_out_of_bounds_part"] == -0.10
    assert "terminal_failure_part" not in result
    assert result["final_score"] == pytest.approx(0.49)


def test_exceed_height_penalty_can_be_weighted_separately_without_changing_default():
    stats = {
        "format_events": ["strict_json"],
        "out_of_bounds_x_count": 1,
        "out_of_bounds_y_count": 1,
        "exceed_height_count": 2,
        "total_steps": 4,
        "total_object_volume": 1.0,
        "total_object_count": 2,
        "termination_reason": "max_invalid_actions",
    }

    default_result = _scorer().compute_final_score_with_info(_FakeEnv(), stats)
    assert default_result["xyz_out_of_bounds_count"] == 4
    assert default_result["out_of_bounds_x_count"] == 1
    assert default_result["out_of_bounds_y_count"] == 1
    assert default_result["exceed_height_count"] == 2
    assert default_result["out_of_bounds_xy_part"] == pytest.approx(-0.2)
    assert default_result["exceed_height_part"] == pytest.approx(-0.2)
    assert default_result["xyz_out_of_bounds_part"] == pytest.approx(-0.4)
    assert default_result["final_score"] == pytest.approx(0.25)

    cfg = _scorer().cfg
    cfg.failure.exceed_height_score = -0.20
    weighted_result = PackingScore(cfg).compute_final_score_with_info(_FakeEnv(), stats)
    assert weighted_result["out_of_bounds_xy_part"] == pytest.approx(-0.2)
    assert weighted_result["exceed_height_part"] == pytest.approx(-0.4)
    assert weighted_result["xyz_out_of_bounds_part"] == pytest.approx(-0.6)
    assert weighted_result["final_score"] == pytest.approx(0.05)
