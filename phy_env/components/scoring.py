"""Scoring components for compactness and final packing quality.

- Each compactness metric returns one score in [0, 1].
- WeightedScore combines configured metrics and handles granularity and infeasible penalties.
"""

from abc import ABC, abstractmethod

import numpy as np
from omegaconf import OmegaConf

from phy_env.metrics import final_packing_metrics
from phy_env.registry import register_component


class BaseCompactness(ABC):
    """Base class for compactness metrics over the current environment state."""

    @abstractmethod
    def score(self, env) -> float:
        ...

    @staticmethod
    def _total_object_volume(placed):
        return sum(l * w * h for (l, w, h) in (o["size_m"] for o in placed))


@register_component("compactness", "height_norm_compactness")
class HeightNormCompactness(BaseCompactness):
    """Compactness normalized by full container floor area times max occupied height."""

    def __init__(self, cfg=None):
        pass

    def score(self, env) -> float:
        placed = env.get_placed_objects()
        if not placed:
            return 0.0
        total_volume = self._total_object_volume(placed)
        max_height = env.get_occupied_height()
        l_cells, w_cells, _ = env.container_size
        base_area = (l_cells * env.xy_resolution) * (w_cells * env.xy_resolution)
        bbox_volume = base_area * max_height
        if bbox_volume <= 1e-9:
            return 0.0
        return float(np.clip(total_volume / bbox_volume, 0.0, 1.0))


@register_component("compactness", "bbox_compactness")
class BboxCompactness(BaseCompactness):
    """Compactness normalized by occupied XY bounding-box area times max occupied height."""

    def __init__(self, cfg=None):
        pass

    def score(self, env) -> float:
        placed = env.get_placed_objects()
        if not placed:
            return 0.0
        total_volume = self._total_object_volume(placed)
        # Occupied XY bounding box derived from non-empty height-map cells.
        height_map, _ = env.scan_height_map()
        occupied = height_map > 0
        if not np.any(occupied):
            return 0.0
        xs = np.any(occupied, axis=1)
        ys = np.any(occupied, axis=0)
        nx = int(np.flatnonzero(xs)[-1] - np.flatnonzero(xs)[0] + 1)
        ny = int(np.flatnonzero(ys)[-1] - np.flatnonzero(ys)[0] + 1)
        max_height = float(height_map.max())
        bbox_volume = (nx * env.xy_resolution) * (ny * env.xy_resolution) * max_height
        if bbox_volume <= 1e-9:
            return 0.0
        return float(np.clip(total_volume / bbox_volume, 0.0, 1.0))


@register_component("scoring", "weighted")
class WeightedScore:
    """Combine configured compactness metrics with weights.

    Example config:
        type: weighted
        granularity: final          # final | per_step
        infeasible_penalty: 0.0
        components:
          - type: bbox_compactness
            weight: 1.0
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.granularity = cfg.get("granularity", "final")
        self.infeasible_penalty = float(cfg.get("infeasible_penalty", 0.0))
        from phy_env.registry import build_component
        self._items = []  # [(component, weight), ...]
        for c in cfg.get("components", []):
            comp = build_component("compactness", c)
            self._items.append((comp, float(c.get("weight", 1.0))))

    def _weighted_score(self, env) -> float:
        return sum(w * comp.score(env) for comp, w in self._items)

    def compute_step_score(self, env, action, valid):
        # Final-only scoring does not emit per-step scores.
        if self.granularity == "final":
            return 0.0
        return self._weighted_score(env)

    def compute_final_score(self, env, termination_reason):
        score = self._weighted_score(env)
        if termination_reason == "infeasible":
            score += self.infeasible_penalty
        return score


@register_component("scoring", "packing_score")
class PackingScore:
    """Eval-aligned final packing score with format and xyz out-of-bounds penalties."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.granularity = cfg.get("granularity", "final")
        self._compactness = None
        if "compactness" in cfg:
            from phy_env.registry import build_component

            self._compactness = build_component("compactness", OmegaConf.create({"type": cfg.compactness.type}))

    def compute_step_score(self, env, action, valid):
        return 0.0

    def compute_final_score(self, env, termination_reason):
        return self.compute_final_score_with_info(env, {"termination_reason": termination_reason})["final_score"]

    def compute_final_score_with_info(self, env, episode_stats):
        total_object_volume = float(episode_stats.get("total_object_volume", 0.0) or 0.0)
        total_object_count = int(episode_stats.get("total_object_count", 0) or 0)
        if total_object_volume <= 0.0 or total_object_count <= 0:
            placed = env.get_placed_objects()
            total_object_volume = sum(float(np.prod(obj["size_m"])) for obj in placed)
            total_object_count = len(placed)
        metrics = final_packing_metrics(env, total_object_volume, total_object_count)
        compactness_raw = float(metrics["center_xyz_inside_container_compactness"])

        quality_cfg = self.cfg.get("quality", {})
        quality_inside_volume_part = 0.0
        quality_compactness_part = 0.0
        if quality_cfg.get("enable", True):
            quality_inside_volume_part = (
                float(quality_cfg.get("inside_volume_ratio_weight", 0.7))
                * float(metrics["center_xyz_inside_volume_ratio"])
            )
            quality_compactness_part = (
                float(quality_cfg.get("inside_container_compactness_weight", 0.3))
                * float(metrics["center_xyz_inside_container_compactness"])
            )

        format_part = 0.0
        events = episode_stats.get("format_events", [])
        if self.cfg.format.get("enable", True) and events:
            scores = []
            for event in events:
                if event == "strict_json":
                    scores.append(float(self.cfg.format.strict_json_score))
                elif event == "extra_text":
                    scores.append(float(self.cfg.format.extra_text_score))
                else:
                    scores.append(float(self.cfg.format.parse_fail_score))
            format_part = float(self.cfg.format.weight) * float(np.mean(scores))

        failure_part = 0.0
        has_split_oob_counts = any(
            key in episode_stats for key in ("out_of_bounds_x_count", "out_of_bounds_y_count", "exceed_height_count")
        )
        out_of_bounds_x_count = int(episode_stats.get("out_of_bounds_x_count", 0) or 0)
        out_of_bounds_y_count = int(episode_stats.get("out_of_bounds_y_count", 0) or 0)
        exceed_height_count = int(episode_stats.get("exceed_height_count", 0) or 0)
        if has_split_oob_counts:
            xyz_out_of_bounds_count = out_of_bounds_x_count + out_of_bounds_y_count + exceed_height_count
        else:
            xyz_out_of_bounds_count = int(episode_stats.get("xyz_out_of_bounds_count", 0) or 0)
            out_of_bounds_x_count = 0
            out_of_bounds_y_count = 0
            exceed_height_count = 0
        xyz_out_of_bounds_part = 0.0
        out_of_bounds_xy_part = 0.0
        exceed_height_part = 0.0
        if self.cfg.failure.get("enable", True):
            xyz_out_of_bounds_score = float(self.cfg.failure.xyz_out_of_bounds_score)
            exceed_height_score = float(self.cfg.failure.get("exceed_height_score", xyz_out_of_bounds_score))
            if has_split_oob_counts:
                out_of_bounds_xy_part = (out_of_bounds_x_count + out_of_bounds_y_count) * xyz_out_of_bounds_score
                exceed_height_part = exceed_height_count * exceed_height_score
                xyz_out_of_bounds_part = out_of_bounds_xy_part + exceed_height_part
            else:
                xyz_out_of_bounds_part = xyz_out_of_bounds_count * xyz_out_of_bounds_score
            failure_part = xyz_out_of_bounds_part

        final_score = quality_inside_volume_part + quality_compactness_part + format_part + failure_part
        return {
            **metrics,
            "compactness_raw": compactness_raw,
            "quality_inside_volume_part": quality_inside_volume_part,
            "quality_compactness_part": quality_compactness_part,
            "format_part": format_part,
            "xyz_out_of_bounds_count": xyz_out_of_bounds_count,
            "out_of_bounds_x_count": out_of_bounds_x_count,
            "out_of_bounds_y_count": out_of_bounds_y_count,
            "exceed_height_count": exceed_height_count,
            "out_of_bounds_xy_part": out_of_bounds_xy_part,
            "exceed_height_part": exceed_height_part,
            "xyz_out_of_bounds_part": xyz_out_of_bounds_part,
            "failure_part": failure_part,
            "final_score": final_score,
        }
