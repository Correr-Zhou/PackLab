"""Constructed layer-cuboid episode generator.

The generator builds a perfect stacked layout first and derives object sizes
and ground-truth actions from that layout. It intentionally keeps the first
version narrow so replay can validate every generated episode.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


XY_CELL_CM = 2.5
Z_CELL_CM = 0.2


DIFFICULTY_PROFILES = {
    "easy": {
        "object_count": (4, 8),
        "layers": (1, 2),
        "fill_height_ratio": (0.45, 0.70),
        "buckets": [(50, 50, 40), (60, 50, 45), (60, 60, 50), (80, 60, 55)],
    },
    "medium": {
        "object_count": (9, 18),
        "layers": (2, 3),
        "fill_height_ratio": (0.45, 0.70),
        "buckets": [(80, 80, 60), (100, 80, 70), (100, 100, 80), (120, 100, 90)],
    },
    "hard": {
        "object_count": (19, 32),
        "layers": (3, 4),
        "fill_height_ratio": (0.84, 0.96),
        "buckets": [(120, 120, 90), (140, 120, 100), (140, 140, 100), (160, 120, 110)],
    },
}


@dataclass
class GenerationConfig:
    num_episodes: int
    difficulties: dict[str, float]
    buffer_sizes: list[int]
    seed: int = 0
    xy_cell_cm: float = XY_CELL_CM
    z_cell_cm: float = Z_CELL_CM
    min_xy_side_cm: float = 7.5
    max_xy_side_cm: float = 170.0
    max_area_ratio: float = 0.35
    max_aspect_ratio: float = 3.0
    min_height_cm: float = 8.0
    max_height_cm: float = 56.0
    container_xy_jitter_cm: list[float] = field(
        default_factory=lambda: [i * XY_CELL_CM for i in range(-4, 5)]
    )
    container_z_jitter_cm: list[float] = field(default_factory=lambda: [float(i) for i in range(-10, 11)])
    first_anchor_distribution: dict[str, float] = field(default_factory=dict)
    order_strategy: str = "layer_random_anchor"


class LayerCuboidGenerator:
    def __init__(self, cfg: GenerationConfig):
        self.cfg = cfg
        self._base_rng = random.Random(cfg.seed)
        total = sum(float(v) for v in cfg.difficulties.values())
        if total <= 0:
            raise ValueError("difficulty weights must sum to a positive value")
        self._difficulty_items = [(k, float(v) / total) for k, v in cfg.difficulties.items()]

    def generate(self) -> list[dict]:
        difficulties = self._expanded_difficulty_schedule()
        buffer_sizes = self._expanded_buffer_schedule()
        return [
            self.generate_episode(i, difficulty=difficulties[i], buffer_size=buffer_sizes[i])
            for i in range(self.cfg.num_episodes)
        ]

    def generate_episode(self, index: int, difficulty: str | None = None, buffer_size: int | None = None) -> dict:
        rng = random.Random(self.cfg.seed + index * 1009)
        difficulty = difficulty or self._sample_difficulty(rng)
        profile = DIFFICULTY_PROFILES[difficulty]
        container_size_cm, bucket_name = self._sample_container(rng, difficulty, profile)
        x_cells = int(round(container_size_cm[0] / self.cfg.xy_cell_cm))
        y_cells = int(round(container_size_cm[1] / self.cfg.xy_cell_cm))
        z_cells = int(round(container_size_cm[2] / self.cfg.z_cell_cm))
        object_target = rng.randint(*profile["object_count"])
        layer_count = self._sample_layer_count(rng, profile, object_target)
        layer_heights = self._sample_layer_heights(rng, profile, container_size_cm[2], layer_count)
        counts = self._split_count_across_layers(rng, object_target, layer_count)

        objects = []
        gt_plan = []
        z_cursor = 0
        obj_idx = 0
        for layer_idx, (height_cm, count) in enumerate(zip(layer_heights, counts)):
            rects = self._split_layer(rng, x_cells, y_cells, count)
            if self.cfg.order_strategy == "layer_random_anchor":
                rects = self._order_layer_rects(rng, rects, x_cells, y_cells, prefer_anchor=layer_idx == 0)
            dz = int(round(height_cm / self.cfg.z_cell_cm))
            for rect in rects:
                x, y, dx, dy = rect
                rotation = 1 if rng.random() < 0.25 and dx != dy else 0
                if rotation:
                    size_cells = [dy, dx, dz]
                else:
                    size_cells = [dx, dy, dz]
                size_cm = [
                    round(size_cells[0] * self.cfg.xy_cell_cm, 4),
                    round(size_cells[1] * self.cfg.xy_cell_cm, 4),
                    round(size_cells[2] * self.cfg.z_cell_cm, 4),
                ]
                internal_id = f"obj_{obj_idx:05d}"
                obj = {
                    "internal_id": internal_id,
                    "gt_index": obj_idx,
                    "type_id": obj_idx % 20,
                    "size_cm": size_cm,
                }
                objects.append(obj)
                gt_plan.append({
                    "internal_id": internal_id,
                    "gt_index": obj_idx,
                    "layer_index": layer_idx,
                    "target_box_cells": {"x": x, "y": y, "z": z_cursor, "dx": dx, "dy": dy, "dz": dz},
                    "action": {"object_id": None, "x": x, "y": y, "rotation": rotation},
                })
                obj_idx += 1
            z_cursor += dz

        seed = self.cfg.seed + index
        if self.cfg.order_strategy not in {"layer_random_anchor", "legacy"}:
            raise ValueError(f"invalid order_strategy: {self.cfg.order_strategy}")
        return {
            "episode_id": f"{difficulty}_{index:06d}",
            "sample_id": f"{difficulty}_{index:06d}",
            "difficulty": difficulty,
            "container_bucket": bucket_name,
            "container_size_cm": container_size_cm,
            "container_grid_size": {"x_cells": x_cells, "y_cells": y_cells, "z_cells": z_cells},
            "grid_cell_size_cm": {"xy": self.cfg.xy_cell_cm, "z": self.cfg.z_cell_cm},
            "buffer_size": int(buffer_size if buffer_size is not None else rng.choice(self.cfg.buffer_sizes)),
            "object_sequence": objects,
            "gt_plan": gt_plan,
            "seed": seed,
            "generator": f"layer_cuboid_v1_{self.cfg.order_strategy}",
        }

    def _expanded_difficulty_schedule(self) -> list[str]:
        names = [name for name, _ in self._difficulty_items]
        raw = {name: self.cfg.num_episodes * dict(self._difficulty_items)[name] for name in names}
        counts = {name: int(raw[name]) for name in names}
        remaining = self.cfg.num_episodes - sum(counts.values())
        remainders = sorted(((raw[name] - counts[name], name) for name in names), reverse=True)
        for _, name in remainders[:remaining]:
            counts[name] += 1
        schedule = []
        for name in names:
            schedule.extend([name] * counts[name])
        rng = random.Random(self.cfg.seed + 17)
        rng.shuffle(schedule)
        return schedule

    def _expanded_buffer_schedule(self) -> list[int]:
        sizes = [int(v) for v in self.cfg.buffer_sizes]
        counts = {size: self.cfg.num_episodes // len(sizes) for size in sizes}
        for size in sizes[: self.cfg.num_episodes - sum(counts.values())]:
            counts[size] += 1
        schedule = []
        for size in sizes:
            schedule.extend([size] * counts[size])
        rng = random.Random(self.cfg.seed + 31)
        rng.shuffle(schedule)
        return schedule

    def _sample_difficulty(self, rng: random.Random) -> str:
        value = rng.random()
        cumulative = 0.0
        for name, weight in self._difficulty_items:
            cumulative += weight
            if value <= cumulative:
                return name
        return self._difficulty_items[-1][0]

    def _sample_container(self, rng: random.Random, difficulty: str, profile: dict) -> tuple[list[float], str]:
        base = rng.choice(profile["buckets"])
        x = self._clamp(base[0] + rng.choice(self.cfg.container_xy_jitter_cm), 50, 170)
        y = self._clamp(base[1] + rng.choice(self.cfg.container_xy_jitter_cm), 50, 170)
        z = self._clamp(base[2] + rng.choice(self.cfg.container_z_jitter_cm), 35, 120)
        x = round(round(x / self.cfg.xy_cell_cm) * self.cfg.xy_cell_cm, 4)
        y = round(round(y / self.cfg.xy_cell_cm) * self.cfg.xy_cell_cm, 4)
        z = round(round(z / 1.0) * 1.0, 4)
        return [x, y, z], f"{difficulty}_{base[0]}x{base[1]}x{base[2]}"

    def _sample_layer_count(self, rng: random.Random, profile: dict, object_target: int) -> int:
        lo, hi = profile["layers"]
        hi = min(hi, object_target)
        return rng.randint(lo, hi)

    def _sample_layer_heights(self, rng: random.Random, profile: dict, container_h_cm: float, layer_count: int) -> list[float]:
        ratio = rng.uniform(*profile["fill_height_ratio"])
        target_h = min(container_h_cm * ratio, container_h_cm - 1.0)
        target_h = max(target_h, layer_count * self.cfg.min_height_cm)
        weights = [rng.uniform(0.75, 1.25) for _ in range(layer_count)]
        raw = [target_h * w / sum(weights) for w in weights]
        heights = [self._quantize(max(self.cfg.min_height_cm, min(self.cfg.max_height_cm, h)), 1.0) for h in raw]
        total = sum(heights)
        if total > container_h_cm - 1.0:
            scale = (container_h_cm - 1.0) / total
            heights = [self._quantize(max(self.cfg.min_height_cm, h * scale), 1.0) for h in heights]
        return heights

    def _split_count_across_layers(self, rng: random.Random, object_target: int, layer_count: int) -> list[int]:
        counts = [1] * layer_count
        for _ in range(object_target - layer_count):
            counts[rng.randrange(layer_count)] += 1
        rng.shuffle(counts)
        return counts

    def _split_layer(self, rng: random.Random, x_cells: int, y_cells: int, target_count: int) -> list[tuple[int, int, int, int]]:
        rects = [(0, 0, x_cells, y_cells)]
        attempts = 0
        while len(rects) < target_count and attempts < target_count * 200:
            attempts += 1
            idx = self._pick_splittable_rect(rng, rects)
            if idx is None:
                break
            rect = rects.pop(idx)
            split = self._try_split_rect(rng, rect, x_cells, y_cells)
            if split is None:
                rects.append(rect)
                continue
            rects.extend(split)
        if len(rects) != target_count:
            raise RuntimeError(f"failed to split layer into {target_count} rectangles")
        return rects

    def _order_layer_rects(
        self,
        rng: random.Random,
        rects: list[tuple[int, int, int, int]],
        x_cells: int,
        y_cells: int,
        prefer_anchor: bool,
    ) -> list[tuple[int, int, int, int]]:
        ordered = list(rects)
        rng.shuffle(ordered)
        if not prefer_anchor:
            return ordered
        anchor = self._sample_first_anchor(rng)
        ranked = sorted(
            enumerate(ordered),
            key=lambda item: self._anchor_rank(item[1], x_cells, y_cells, anchor, item[0]),
        )
        first_idx = ranked[0][0]
        first = ordered.pop(first_idx)
        return [first] + ordered

    def _sample_first_anchor(self, rng: random.Random) -> str:
        distribution = self.cfg.first_anchor_distribution or {
            "top_left": 0.225,
            "top_right": 0.225,
            "bottom_left": 0.225,
            "bottom_right": 0.225,
            "edge_or_interior": 0.10,
        }
        total = sum(float(v) for v in distribution.values())
        if total <= 0:
            return "top_left"
        value = rng.random() * total
        cumulative = 0.0
        for name, weight in distribution.items():
            cumulative += float(weight)
            if value <= cumulative:
                return str(name)
        return str(next(reversed(distribution)))

    @staticmethod
    def _anchor_rank(
        rect: tuple[int, int, int, int],
        x_cells: int,
        y_cells: int,
        anchor: str,
        tie_breaker: int,
    ) -> tuple[int, int, int, int]:
        x, y, dx, dy = rect
        max_x = x_cells - dx
        max_y = y_cells - dy
        if anchor == "top_left":
            return (abs(x), abs(y), x + y, tie_breaker)
        if anchor == "top_right":
            return (abs(x), abs(y - max_y), x + abs(y - max_y), tie_breaker)
        if anchor == "bottom_left":
            return (abs(x - max_x), abs(y), abs(x - max_x) + y, tie_breaker)
        if anchor == "bottom_right":
            return (abs(x - max_x), abs(y - max_y), abs(x - max_x) + abs(y - max_y), tie_breaker)
        edge_distance = min(x, y, max_x - x, max_y - y)
        center_distance = abs((x + dx / 2) - x_cells / 2) + abs((y + dy / 2) - y_cells / 2)
        return (edge_distance, -int(center_distance), x + y, tie_breaker)

    def _pick_splittable_rect(self, rng: random.Random, rects: list[tuple[int, int, int, int]]) -> int | None:
        candidates = [i for i, rect in enumerate(rects) if self._can_split(rect)]
        if not candidates:
            return None
        return rng.choice(candidates)

    def _can_split(self, rect: tuple[int, int, int, int]) -> bool:
        _, _, dx, dy = rect
        min_cells = int(round(self.cfg.min_xy_side_cm / self.cfg.xy_cell_cm))
        return dx >= min_cells * 2 or dy >= min_cells * 2

    def _try_split_rect(
        self, rng: random.Random, rect: tuple[int, int, int, int], layer_x_cells: int, layer_y_cells: int
    ) -> list[tuple[int, int, int, int]] | None:
        x, y, dx, dy = rect
        axes = ["x", "y"] if dx >= dy else ["y", "x"]
        if rng.random() < 0.25:
            axes.reverse()
        for axis in axes:
            min_cells = int(round(self.cfg.min_xy_side_cm / self.cfg.xy_cell_cm))
            span = dx if axis == "x" else dy
            if span < min_cells * 2:
                continue
            lo = min_cells
            hi = span - min_cells
            valid_points = list(range(lo, hi + 1))
            rng.shuffle(valid_points)
            for cut in valid_points:
                if axis == "x":
                    children = [(x, y, cut, dy), (x + cut, y, dx - cut, dy)]
                else:
                    children = [(x, y, dx, cut), (x, y + cut, dx, dy - cut)]
                if all(self._rect_ok(child, layer_x_cells, layer_y_cells) for child in children):
                    return children
        return None

    def _rect_ok(self, rect: tuple[int, int, int, int], layer_x_cells: int, layer_y_cells: int) -> bool:
        _, _, dx, dy = rect
        min_cells = int(round(self.cfg.min_xy_side_cm / self.cfg.xy_cell_cm))
        max_cells = int(round(self.cfg.max_xy_side_cm / self.cfg.xy_cell_cm))
        if dx < min_cells or dy < min_cells or dx > max_cells or dy > max_cells:
            return False
        if max(dx / dy, dy / dx) > self.cfg.max_aspect_ratio:
            return False
        return True

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return min(hi, max(lo, value))

    @staticmethod
    def _quantize(value: float, step: float) -> float:
        return round(round(value / step) * step, 4)
