"""PyBullet packing environment with a Gym-like interface.

The environment is sample-driven: container_size, buffer_size, and
object_sequence are all provided by each sample. Visible buffer objects are
represented as real rigid bodies on a gray staging slab, then moved into the
container during placement.
"""

import random
import time
from dataclasses import dataclass, field

import numpy as np

from phy_env.registry import build_component
from phy_env.render_utils import (
    apply_ground_background,
    depth_background_mask,
    enhance_rgb,
    project_bbox_ndc_center,
    resize_mask_nearest,
    resize_rgb_lanczos,
    segmentation_body_mask,
)

# Object color palette, indexed by type_id.
MORANDI_OBJECT_PALETTE = [
    (0.385, 0.476, 0.525, 1.0), (0.459, 0.533, 0.451, 1.0), (0.558, 0.492, 0.574, 1.0),
    (0.607, 0.517, 0.410, 1.0), (0.476, 0.443, 0.385, 1.0), (0.369, 0.500, 0.476, 1.0),
    (0.525, 0.476, 0.410, 1.0), (0.426, 0.467, 0.574, 1.0), (0.574, 0.541, 0.459, 1.0),
    (0.451, 0.541, 0.574, 1.0), (0.508, 0.451, 0.508, 1.0), (0.410, 0.492, 0.410, 1.0),
    (0.599, 0.558, 0.508, 1.0), (0.402, 0.435, 0.500, 1.0), (0.517, 0.558, 0.492, 1.0),
    (0.574, 0.508, 0.476, 1.0), (0.476, 0.525, 0.558, 1.0), (0.541, 0.500, 0.426, 1.0),
    (0.435, 0.508, 0.517, 1.0), (0.566, 0.533, 0.574, 1.0),
]


@dataclass
class Observation:
    image: np.ndarray = None
    buffer_objects: list = field(default_factory=list)
    step_index: int = 0
    done: bool = False


@dataclass
class StepResult:
    observation: Observation = None
    score: float = 0.0
    done: bool = False
    info: dict = field(default_factory=dict)


def _morandi_color(index):
    return list(MORANDI_OBJECT_PALETTE[index % len(MORANDI_OBJECT_PALETTE)])


class BufferWindow:
    """Read-only sliding window over a sample object sequence."""

    def __init__(self, object_sequence, buffer_size, shuffle_visible_objects=False, seed=None):
        self._sequence = list(object_sequence)
        self._buffer_size = int(buffer_size)
        self._shuffle_visible_objects = bool(shuffle_visible_objects)
        self._rng = random.Random(seed)
        self._next_idx = 0
        self._window = []
        self._visible_order = []
        self._fill()
        self._refresh_visible_order()

    def _fill(self):
        while len(self._window) < self._buffer_size and self._next_idx < len(self._sequence):
            obj = dict(self._sequence[self._next_idx])
            obj["seq_index"] = self._next_idx
            self._window.append(obj)
            self._next_idx += 1

    def _refresh_visible_order(self):
        self._visible_order = list(range(len(self._window)))
        if self._shuffle_visible_objects:
            self._rng.shuffle(self._visible_order)

    def get_visible_objects(self):
        visible = []
        for visible_id, window_idx in enumerate(self._visible_order):
            obj = dict(self._window[window_idx])
            obj["object_id"] = visible_id
            obj["buffer_id"] = visible_id
            obj["window_index"] = window_idx
            visible.append(obj)
        return visible

    def advance(self, picked_buffer_id):
        window_idx = self._visible_order[picked_buffer_id]
        self._window.pop(window_idx)
        self._fill()
        self._refresh_visible_order()

    def is_exhausted(self):
        return self._next_idx >= len(self._sequence) and len(self._window) == 0

    def num_visible(self):
        return len(self._window)


class PackingEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        ecfg = cfg.env
        self.hz = int(ecfg.hz)
        self.settle_steps = int(ecfg.settle_steps)
        # Run at least settle_steps, then continue until objects are still or the step cap is reached.
        self.max_settle_steps = int(ecfg.get("max_settle_steps", self.settle_steps * 4))
        self.settle_velocity_threshold = float(ecfg.get("settle_velocity_threshold", 0.01))
        self.render_enabled = bool(ecfg.render)
        # Termination controls.
        self.enable_early_stop = bool(ecfg.termination.enable_early_stop)
        self.max_consecutive_fails = int(ecfg.termination.max_consecutive_fails)
        # Shared container discretization. The physical container size comes from each sample, in cm.
        ccfg = cfg.env.container
        # Convert grid cell sizes from cm to meters for PyBullet.
        self.xy_cell_cm = float(ccfg.xy_cell_cm)
        self.z_cell_cm = float(ccfg.z_cell_cm)
        self.xy_resolution = self.xy_cell_cm / 100.0
        self.z_resolution = self.z_cell_cm / 100.0
        self.show_wall = bool(ccfg.show_wall)
        self.height_map_color_norm_container_ratio = float(ccfg.get("height_map_color_norm_container_ratio", 2.0))
        self.wall_width = 0.005
        self.wall_color = (1.0, 1.0, 1.0, 0.0)
        # Fixed scene layout parameters.
        self.target_origin = (0.30, -0.35, 0.0)
        self.plane_offset = (0.0, 0.0, 0.0)
        self.render_width = 1280
        self.render_height = 960
        self.render_aa_scale = int(ecfg.get("render_aa_scale", 1))
        # Buffer-area layout parameters.
        self.buffer_gap_x = 0.5
        self.buffer_offset_y = -1.2
        self.buffer_spacing = 0.08
        self.buffer_grid_cols = 0
        self.shuffle_visible_objects = bool(ecfg.get("buffer", {}).get("shuffle_visible_objects", False))
        # Configurable components.
        self.observation = build_component("observation", cfg.observation)
        self.policy = build_component("policy", cfg.policy)
        self.scoring = build_component("scoring", cfg.scoring)
        # Runtime state.
        self.p = None
        self.client_id = None
        self.container_size = None
        self.buffer = None
        self.placed_objects = []
        self.container_wall_body_ids = []
        self.container_wall_specs = []
        # Physical buffer objects, aligned with the visible buffer window.
        self.buffered_objects = []
        self.buffered_phy_sim_obj_ids = []
        # Optional visualization callback. Training keeps this unset to avoid overhead.
        self.record_callback = None
        self.record_interval = 24
        self.step_index = 0
        self.consecutive_fails = 0
        self.done = False
        self.termination_reason = None

    # ---------------- Lifecycle ----------------

    def reset(self, sample):
        # Parse sample metadata. Container dimensions are stored in cm and converted to grid cells.
        csz_cm = sample["container_size_cm"]
        self.container_size = (
            int(round(csz_cm[0] / self.xy_cell_cm)),
            int(round(csz_cm[1] / self.xy_cell_cm)),
            int(round(csz_cm[2] / self.z_cell_cm)),
        )
        self.buffer_size = int(sample["buffer_size"])
        seq = sample["object_sequence"]
        seed = sample.get("seed", None)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        # Attach physical properties, including color and meter-scale size.
        prepared = []
        for obj in seq:
            size_cm = obj["size_cm"]
            prepared.append({
                "internal_id": obj.get("internal_id", f"obj_{len(prepared):05d}"),
                "gt_index": obj.get("gt_index", len(prepared)),
                "type_id": int(obj["type_id"]),
                "size_m": (size_cm[0] / 100.0, size_cm[1] / 100.0, size_cm[2] / 100.0),
                "color": _morandi_color(int(obj["type_id"])),
            })
        self._object_sequence = prepared
        self.buffer = BufferWindow(
            prepared, self.buffer_size, shuffle_visible_objects=self.shuffle_visible_objects, seed=seed
        )
        # Reset the physics world.
        self._start_pybullet()
        self._reset_scene()
        self._add_gray_buffer_area()
        self._new_container()
        # Cache the camera matrices once per reset and reuse them for captures.
        self._compute_camera_matrices()
        # Reset episode state.
        self.placed_objects = []
        self.buffered_objects = []
        self.buffered_phy_sim_obj_ids = []
        self.step_index = 0
        self.consecutive_fails = 0
        self.done = False
        self.termination_reason = None
        # Place visible buffer objects as rigid bodies on staging slots.
        self._prepare_buffered_objects()
        return self.observe()

    def observe(self):
        image = self.observation.render(self)
        return Observation(
            image=image,
            buffer_objects=self.buffer.get_visible_objects(),
            step_index=self.step_index,
            done=self.done,
        )

    def step(self, action):
        valid, reason = self._try_place(action)
        if valid:
            self.consecutive_fails = 0
        else:
            self.consecutive_fails += 1
        self.step_index += 1
        # Refill physical buffer bodies after a successful placement.
        if valid:
            self._prepare_buffered_objects()
        self._update_done(valid)
        score = self.scoring.compute_step_score(self, action, valid)
        info = {"valid": valid, "reason": reason, "termination_reason": self.termination_reason}
        return StepResult(observation=self.observe(), score=score, done=self.done, info=info)

    def step_relaxed(self, action, force_outside=False, outside_seed=None):
        valid, reason = self._try_place_relaxed(action, force_outside=force_outside, outside_seed=outside_seed)
        if valid:
            self.consecutive_fails = 0
        else:
            self.consecutive_fails += 1
        self.step_index += 1
        self._prepare_buffered_objects()
        if self.buffer.is_exhausted():
            self.done = True
            self.termination_reason = "completed"
        score = self.scoring.compute_step_score(self, action, valid)
        info = {"valid": valid, "reason": reason, "termination_reason": self.termination_reason}
        return StepResult(observation=self.observe(), score=score, done=self.done, info=info)

    def get_final_score(self):
        reason = self.termination_reason or "completed"
        return self.scoring.compute_final_score(self, reason)

    def close(self):
        if self.p is not None and self.p.isConnected():
            self.p.disconnect()
            self.p = None

    # ---------------- Termination ----------------

    def _update_done(self, last_valid):
        if self.buffer.is_exhausted():
            self.done = True
            self.termination_reason = "completed"
            return
        if self.enable_early_stop and self.consecutive_fails >= self.max_consecutive_fails:
            self.done = True
            self.termination_reason = "infeasible"

    # ---------------- Physical Buffer Objects ----------------

    def _create_box_shape(self, size, color):
        half = [size[0] / 2.0, size[1] / 2.0, size[2] / 2.0]
        col = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=half)
        vis = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        return col, vis

    def _add_gray_buffer_area(self):
        # Size slots from the largest object in the current sample sequence.
        max_length_m = max(o["size_m"][0] for o in self._object_sequence)
        max_width_m = max(o["size_m"][1] for o in self._object_sequence)
        cols = self.buffer_grid_cols if self.buffer_grid_cols > 0 else int(np.ceil(np.sqrt(self.buffer_size)))
        self._buf_cols = max(1, min(cols, self.buffer_size))
        self._buf_rows = int(np.ceil(self.buffer_size / self._buf_cols))
        self._buf_cell_l = max_length_m
        self._buf_cell_w = max_width_m
        self._buf_padding = max(0.32, self.buffer_spacing * 3.5)
        slot_l = self._buf_cell_l * self._buf_cols + self.buffer_spacing * (self._buf_cols - 1)
        slot_w = self._buf_cell_w * self._buf_rows + self.buffer_spacing * (self._buf_rows - 1)
        area_l = slot_l + self._buf_padding * 2.0
        area_w = slot_w + self._buf_padding * 2.0
        red_left_x = self.target_origin[0]
        red_front_y = self.target_origin[1]
        buffer_front_y = red_front_y + self.buffer_offset_y
        buffer_right_x = red_left_x - self.buffer_gap_x
        corner_x = buffer_right_x - area_l
        corner_y = buffer_front_y
        bottom_clearance = 0.020
        thickness = 0.030
        self.buffer_surface_z = bottom_clearance + thickness
        center = np.array([corner_x + area_l / 2.0, corner_y + area_w / 2.0, bottom_clearance + thickness / 2.0], dtype=float)
        self.gray_buffer_corner_xy = np.array([corner_x, corner_y], dtype=float)
        self.gray_buffer_bounds_xy = (corner_x, corner_x + area_l, corner_y, corner_y + area_w)
        self._add_box((center, [0, 0, 0, 1]), (area_l, area_w, thickness), (0.78, 0.78, 0.76, 1.0), mass=0)

    def _buffer_slot_corner(self, index, object_size):
        col = index % self._buf_cols
        row = index // self._buf_cols
        x = self.gray_buffer_corner_xy[0] + self._buf_padding + col * (self._buf_cell_l + self.buffer_spacing)
        y = self.gray_buffer_corner_xy[1] + self._buf_padding + row * (self._buf_cell_w + self.buffer_spacing)
        x += max(0.0, (self._buf_cell_l - object_size[0]) / 2.0)
        y += max(0.0, (self._buf_cell_w - object_size[1]) / 2.0)
        return np.array([x, y, 0.0], dtype=float)

    def _prepare_buffered_objects(self):
        # Move existing buffer bodies back to their assigned staging slots.
        visible = self.buffer.get_visible_objects()
        ordered_objects = [dict(obj) for obj in visible]
        body_by_internal_id = {
            obj["internal_id"]: body_id for obj, body_id in zip(self.buffered_objects, self.buffered_phy_sim_obj_ids)
        }
        new_buffered_objects = []
        new_body_ids = []
        for idx, obj in enumerate(ordered_objects):
            size = obj["size_m"]
            corner = self._buffer_slot_corner(idx, size)
            pose = [corner[0] + size[0] / 2.0, corner[1] + size[1] / 2.0, self.buffer_surface_z + size[2] / 2.0]
            body_id = body_by_internal_id.get(obj["internal_id"])
            if body_id is None:
                col, vis = self._create_box_shape(size, obj["color"])
                body_id = self.p.createMultiBody(
                    baseMass=1.0,
                    basePosition=corner + np.array([size[0] / 2.0, size[1] / 2.0, self.buffer_surface_z + size[2] / 2.0]),
                    baseOrientation=[0, 0, 0, 1],
                    baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                )
                self.p.changeVisualShape(body_id, -1, rgbaColor=obj["color"], specularColor=[0.18, 0.18, 0.18])
            else:
                self.p.resetBasePositionAndOrientation(body_id, pose, [0, 0, 0, 1])
            new_buffered_objects.append(obj)
            new_body_ids.append(body_id)
        self.buffered_objects = new_buffered_objects
        self.buffered_phy_sim_obj_ids = new_body_ids

    # ---------------- Placement Execution ----------------

    def _try_place(self, action):
        bid = action.get("object_id")
        if bid is None or bid < 0 or bid >= self.buffer.num_visible():
            return False, "invalid_object_id"
        obj = self.buffered_objects[bid]
        body_id = self.buffered_phy_sim_obj_ids[bid]
        placement = self._action_placement(obj, action)
        rotation = placement["rotation"]
        quat = placement["quat"]
        half_x, half_y, half_z = placement["half_extents"]
        x_center = placement["x_center"]
        y_center = placement["y_center"]
        x_min, x_max, y_min, y_max, bottom_z, top_z = self._container_interior_bounds()
        if not (x_min <= x_center - half_x and x_center + half_x <= x_max):
            eps = 1e-9
            if not (x_min - eps <= x_center - half_x and x_center + half_x <= x_max + eps):
                return False, "out_of_bounds_x"
        if not (y_min <= y_center - half_y and y_center + half_y <= y_max):
            eps = 1e-9
            if not (y_min - eps <= y_center - half_y and y_center + half_y <= y_max + eps):
                return False, "out_of_bounds_y"
        height_map, x_axis, y_axis = self._scan_container_height_map()
        local_h = self._local_max_height(height_map, x_axis, y_axis, x_center, y_center, half_x, half_y)
        z_center = bottom_z + local_h + half_z
        if z_center + half_z > top_z + 1e-9:
            return False, "exceed_height"
        self._place_buffer_body(bid, body_id, obj, rotation, quat, x_center, y_center, z_center)
        return True, None

    def _try_place_relaxed(self, action, force_outside=False, outside_seed=None):
        bid = action.get("object_id")
        if bid is None or bid < 0 or bid >= self.buffer.num_visible():
            return False, "invalid_object_id"
        obj = self.buffered_objects[bid]
        body_id = self.buffered_phy_sim_obj_ids[bid]
        x_min, x_max, y_min, y_max, bottom_z, top_z = self._container_interior_bounds()
        if force_outside:
            rotation = int(action.get("rotation", 0))
            quat = self._rotation_quaternion(rotation)
            oriented = self._oriented_extents(obj["size_m"], quat)
            half_x, half_y, half_z = oriented / 2.0
            x_center, y_center = self._forced_outside_discard_position(outside_seed, half_x, half_y)
            z_center = bottom_z + half_z + 0.01
            reason = "forced_outside"
        else:
            placement = self._action_placement(obj, action)
            rotation = placement["rotation"]
            quat = placement["quat"]
            half_x, half_y, half_z = placement["half_extents"]
            x_center = placement["x_center"]
            y_center = placement["y_center"]
            reasons = self._placement_violation_reasons(x_center, y_center, half_x, half_y, half_z)
            height_map, x_axis, y_axis = self._scan_container_height_map()
            local_h = self._local_max_height(height_map, x_axis, y_axis, x_center, y_center, half_x, half_y)
            z_center = bottom_z + local_h + half_z
            z_center = self._relaxed_spawn_z_center(z_center, reasons, x_center, y_center, half_x, half_y, half_z)
            reason = ",".join(reasons) if reasons else None
        self._place_buffer_body(
            bid, body_id, obj, rotation, quat, x_center, y_center, z_center,
            forced_outside=force_outside,
        )
        return reason is None, reason

    def _action_placement(self, obj, action):
        rotation = int(action.get("rotation", 0))
        quat = self._rotation_quaternion(rotation)
        oriented = self._oriented_extents(obj["size_m"], quat)
        half_x, half_y, half_z = oriented / 2.0
        x0 = self.target_origin[0] + self.wall_width
        y0 = self.target_origin[1] + self.wall_width
        x_center = x0 + int(action["x"]) * self.xy_resolution + half_x
        y_center = y0 + int(action["y"]) * self.xy_resolution + half_y
        return {
            "rotation": rotation,
            "quat": quat,
            "half_extents": (half_x, half_y, half_z),
            "x_center": x_center,
            "y_center": y_center,
        }

    def _placement_violation_reasons(self, x_center, y_center, half_x, half_y, half_z):
        reasons = []
        x_min, x_max, y_min, y_max, bottom_z, top_z = self._container_interior_bounds()
        eps = 1e-9
        if not (x_min - eps <= x_center - half_x and x_center + half_x <= x_max + eps):
            reasons.append("out_of_bounds_x")
        if not (y_min - eps <= y_center - half_y and y_center + half_y <= y_max + eps):
            reasons.append("out_of_bounds_y")
        height_map, x_axis, y_axis = self._scan_container_height_map()
        local_h = self._local_max_height(height_map, x_axis, y_axis, x_center, y_center, half_x, half_y)
        z_center = bottom_z + local_h + half_z
        if z_center + half_z > top_z + eps:
            reasons.append("exceed_height")
        overlap_top_z = self._overlapping_placed_aabb_top_z(x_center, y_center, half_x, half_y)
        if overlap_top_z is not None and overlap_top_z + half_z > top_z + eps and "exceed_height" not in reasons:
            reasons.append("exceed_height")
        return reasons

    def _relaxed_spawn_z_center(self, z_center, reasons, x_center, y_center, half_x, half_y, half_z):
        _, _, _, _, _, top_z = self._container_interior_bounds()
        if "out_of_bounds_x" in reasons or "out_of_bounds_y" in reasons:
            z_center = max(z_center, top_z + half_z)
        overlap_top_z = self._overlapping_placed_aabb_top_z(x_center, y_center, half_x, half_y)
        if overlap_top_z is not None:
            z_center = max(z_center, overlap_top_z + half_z)
        return z_center

    def _overlapping_placed_aabb_top_z(self, x_center, y_center, half_x, half_y):
        x0 = x_center - half_x
        x1 = x_center + half_x
        y0 = y_center - half_y
        y1 = y_center + half_y
        tops = []
        for placed in self.placed_objects:
            aabb_min, aabb_max = self.p.getAABB(placed["body_id"])
            if aabb_max[0] < x0 or aabb_min[0] > x1:
                continue
            if aabb_max[1] < y0 or aabb_min[1] > y1:
                continue
            tops.append(float(aabb_max[2]))
        if not tops:
            return None
        return max(tops)

    def _forced_outside_discard_position(self, outside_seed, half_x, half_y):
        bx0, _, by0, by1 = self.gray_buffer_bounds_xy
        lane_gap = max(self.buffer_spacing, self.xy_resolution * 2.0)
        slot_x = max(2.0 * half_x + lane_gap, lane_gap)
        slot_y = max(2.0 * half_y + lane_gap, lane_gap)
        usable_height = max(0.0, (by1 - by0) - 2.0 * lane_gap)
        rows_per_col = max(1, int(usable_height // slot_y))
        index = sum(1 for placed in self.placed_objects if placed.get("forced_outside"))
        col = index // rows_per_col
        row = index % rows_per_col
        x_center = bx0 - half_x - lane_gap - col * slot_x
        y_top = by1 - half_y - lane_gap
        y_center = y_top - row * slot_y
        y_min_allowed = by0 + half_y + lane_gap
        if y_center < y_min_allowed:
            y_center = (by0 + by1) / 2.0
        return x_center, y_center

    def _place_buffer_body(self, bid, body_id, obj, rotation, quat, x_center, y_center, z_center, forced_outside=False):
        self.p.resetBasePositionAndOrientation(body_id, [x_center, y_center, z_center + 0.01], quat)
        self._simulate_after_insert(active_body_id=body_id)
        pos, _ = self.p.getBasePositionAndOrientation(body_id)
        placed = {
            "body_id": body_id, "type_id": obj["type_id"], "size_m": obj["size_m"],
            "position": pos, "rotation": rotation,
        }
        if forced_outside:
            placed["forced_outside"] = True
        if self._body_center_xyz_outside_container(body_id):
            marker_ids = self._add_outside_marker(body_id)
            placed["outside_marker_body_ids"] = marker_ids
            if forced_outside:
                placed["forced_outside_marker_body_ids"] = marker_ids
        self.placed_objects.append(placed)
        self.buffer.advance(bid)
        self.buffered_objects.pop(bid)
        self.buffered_phy_sim_obj_ids.pop(bid)

    def _body_center_xyz_outside_container(self, body_id):
        pos, _ = self.p.getBasePositionAndOrientation(body_id)
        x_min, x_max, y_min, y_max, bottom_z, top_z = self._container_interior_bounds()
        eps = 1e-9
        return (
            pos[0] < x_min - eps
            or pos[0] > x_max + eps
            or pos[1] < y_min - eps
            or pos[1] > y_max + eps
            or pos[2] < bottom_z - eps
            or pos[2] > top_z + eps
        )

    def _add_outside_marker(self, body_id):
        aabb_min, aabb_max = self.p.getAABB(body_id)
        center = np.array([
            (aabb_min[0] + aabb_max[0]) / 2.0,
            (aabb_min[1] + aabb_max[1]) / 2.0,
            (aabb_min[2] + aabb_max[2]) / 2.0,
        ], dtype=float)
        size = np.array([
            max(0.01, aabb_max[0] - aabb_min[0]),
            max(0.01, aabb_max[1] - aabb_min[1]),
            max(0.01, aabb_max[2] - aabb_min[2]),
        ], dtype=float)
        color = (1.0, 0.82, 0.0, 0.95)
        thickness = max(self.wall_width * 1.5, 0.006)
        return self._add_bbox_edge_markers(center, size, color, thickness)

    def _add_bbox_edge_markers(self, center, size, color, thickness):
        x0 = center[0] - size[0] / 2.0
        x1 = center[0] + size[0] / 2.0
        y0 = center[1] - size[1] / 2.0
        y1 = center[1] + size[1] / 2.0
        z0 = center[2] - size[2] / 2.0
        z1 = center[2] + size[2] / 2.0
        marker_ids = []
        for x in (x0, x1):
            for y in (y0, y1):
                marker_ids.append(self._add_visual_box(([x, y, (z0 + z1) / 2.0], [0, 0, 0, 1]), (thickness, thickness, z1 - z0), color))
        for z in (z0, z1):
            for y in (y0, y1):
                marker_ids.append(self._add_visual_box(([(x0 + x1) / 2.0, y, z], [0, 0, 0, 1]), (x1 - x0, thickness, thickness), color))
            for x in (x0, x1):
                marker_ids.append(self._add_visual_box(([x, (y0 + y1) / 2.0, z], [0, 0, 0, 1]), (thickness, y1 - y0, thickness), color))
        return marker_ids

    def _rotation_quaternion(self, rotation):
        angle = (np.pi / 2.0) if rotation == 1 else 0.0
        return list(self.p.getQuaternionFromEuler([0.0, 0.0, angle]))

    def _oriented_extents(self, extents, quat_xyzw):
        half_local = np.array(extents, dtype=float) / 2.0
        rot = np.array(self.p.getMatrixFromQuaternion(quat_xyzw), dtype=float).reshape(3, 3)
        return (np.abs(rot) @ half_local) * 2.0

    def _simulate_after_insert(self, active_body_id=None):
        # Run the minimum settle steps, then continue until all bodies are still or the cap is reached.
        # If a recorder is attached, capture intermediate falling frames for video output.
        body_ids = [o["body_id"] for o in self.placed_objects]
        if active_body_id is not None:
            body_ids.append(active_body_id)
        for step in range(self.max_settle_steps):
            self.p.stepSimulation()
            if self.record_callback is not None and step % max(1, int(self.record_interval)) == 0:
                self.record_callback(self.capture_scene_rgb())
            if self.render_enabled and self.p.getConnectionInfo(self.client_id).get("connectionMethod") == self.p.GUI:
                time.sleep(1.0 / self.hz)
            # Check stillness only after the minimum settling period, and not on every simulation step.
            if step + 1 >= self.settle_steps and (step + 1) % 24 == 0 and self._all_bodies_settled(body_ids):
                break

    def _all_bodies_settled(self, body_ids):
        # Bodies are considered still when both linear and angular velocities are below the threshold.
        thr = self.settle_velocity_threshold
        for bid in body_ids:
            lin, ang = self.p.getBaseVelocity(bid)
            if max(abs(v) for v in lin) > thr or max(abs(v) for v in ang) > thr:
                return False
        return True

    # ---------------- Component State Accessors ----------------

    def get_placed_objects(self):
        return self.placed_objects

    def get_occupied_height(self):
        height_map, _ = self.scan_height_map()
        return float(height_map.max()) if height_map.size else 0.0

    def scan_height_map(self):
        height_map, _, _ = self._scan_container_height_map()
        return height_map, self._height_map_display_h_max()

    # ---------------- PyBullet Scene Setup ----------------

    def _start_pybullet(self):
        import pybullet as p
        import pybullet_data
        from pybullet_utils import bullet_client

        if self.p is not None and self.p.isConnected():
            self.p.disconnect()
        self.pybullet_data = pybullet_data
        mode = p.GUI if self.render_enabled else p.DIRECT
        self.p = bullet_client.BulletClient(connection_mode=mode)
        self.client_id = self.p._client
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_GUI, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        self.p.configureDebugVisualizer(self.p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self.p.setPhysicsEngineParameter(enableFileCaching=0)
        self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.p.setTimeStep(1.0 / self.hz)
        self.p.setGravity(0, 0, -9.8)

    def _reset_scene(self):
        self.p.resetSimulation(self.p.RESET_USE_DEFORMABLE_WORLD)
        self.p.setGravity(0, 0, -9.8)
        self.ground_body_id = self.p.loadURDF("plane.urdf", self.plane_offset)
        self.container_wall_body_ids = []
        self.container_wall_specs = []

    def _add_box(self, pose, size, color, mass=0):
        pos, orn = pose
        half = [size[0] / 2.0, size[1] / 2.0, size[2] / 2.0]
        col = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=half)
        vis = -1
        if color is not None:
            vis = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        return self.p.createMultiBody(
            baseMass=mass, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=pos, baseOrientation=orn,
        )

    def _add_visual_box(self, pose, size, color):
        pos, orn = pose
        half = [size[0] / 2.0, size[1] / 2.0, size[2] / 2.0]
        vis = self.p.createVisualShape(self.p.GEOM_BOX, halfExtents=half, rgbaColor=color)
        return self.p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
            basePosition=pos, baseOrientation=orn,
        )

    def _new_container(self):
        l, w, h = self.container_size
        block_length = l * self.xy_resolution + self.wall_width * 2
        block_width = w * self.xy_resolution + self.wall_width * 2
        wall_height = h * self.z_resolution
        half_length = block_length / 2.0
        half_width = block_width / 2.0
        half_height = wall_height / 2.0
        tx, ty = self.target_origin[0], self.target_origin[1]
        target_center = np.array([tx + l * self.xy_resolution / 2.0, ty + w * self.xy_resolution / 2.0, 0.0])
        target_center += np.array([self.wall_width, self.wall_width, 0.0])
        container_offset = np.array([0.0, 0.0, 0.025])
        base_color = (0.78, 0.12, 0.08, 1.0)
        edge_color = (1.0, 0.05, 0.02, 0.85)
        wall_specs = [
            (target_center + container_offset + np.array([0.0, half_width, half_height]), (block_length + self.wall_width, self.wall_width, wall_height)),
            (target_center + container_offset - np.array([0.0, half_width, -half_height]), (block_length + self.wall_width, self.wall_width, wall_height)),
            (target_center + container_offset + np.array([half_length, 0.0, half_height]), (self.wall_width, block_width + self.wall_width, wall_height)),
            (target_center + container_offset - np.array([half_length, 0.0, -half_height]), (self.wall_width, block_width + self.wall_width, wall_height)),
        ]
        self.container_wall_specs = [((pos, [0, 0, 0, 1]), size) for pos, size in wall_specs]
        self._restore_container_wall_collisions()
        # Container floor.
        self._add_box((target_center + container_offset, [0, 0, 0, 1]), (block_length, block_width, self.wall_width), base_color, mass=0)
        if self.show_wall:
            self._add_container_edge_markers(target_center + container_offset, block_length, block_width, wall_height, edge_color)

    def _add_container_edge_markers(self, center, length, width, height, color):
        z0 = center[2] + self.wall_width / 2.0
        z1 = center[2] + height
        x0 = center[0] - length / 2.0
        x1 = center[0] + length / 2.0
        y0 = center[1] - width / 2.0
        y1 = center[1] + width / 2.0
        t = self.wall_width * 0.75
        for x in (x0, x1):
            for y in (y0, y1):
                self._add_visual_box(([x, y, (z0 + z1) / 2.0], [0, 0, 0, 1]), (t, t, z1 - z0), color)
        for z in (z0, z1):
            for y in (y0, y1):
                self._add_visual_box(([(x0 + x1) / 2.0, y, z], [0, 0, 0, 1]), (length, t, t), color)
            for x in (x0, x1):
                self._add_visual_box(([x, (y0 + y1) / 2.0, z], [0, 0, 0, 1]), (t, width, t), color)

    def _remove_container_wall_collisions(self):
        for body_id in self.container_wall_body_ids:
            try:
                self.p.removeBody(body_id)
            except Exception:
                pass
        self.container_wall_body_ids = []

    def _restore_container_wall_collisions(self):
        if self.container_wall_body_ids:
            return
        for pose, size in self.container_wall_specs:
            body_id = self._add_box(pose, size, None, mass=0)
            self.p.changeVisualShape(body_id, -1, rgbaColor=self.wall_color)
            self.container_wall_body_ids.append(body_id)

    def _container_interior_bounds(self, margin_x=0.0, margin_y=0.0):
        l_cells, w_cells, z_cells = self.container_size
        origin = np.array(self.target_origin, dtype=float)
        x_min = origin[0] + self.wall_width + margin_x
        x_max = origin[0] + self.wall_width + (l_cells * self.xy_resolution) - margin_x
        y_min = origin[1] + self.wall_width + margin_y
        y_max = origin[1] + self.wall_width + (w_cells * self.xy_resolution) - margin_y
        bottom_z = 0.025 + self.wall_width
        top_z = bottom_z + (z_cells * self.z_resolution)
        return x_min, x_max, y_min, y_max, bottom_z, top_z

    def _height_map_display_h_max(self):
        _, _, _, _, bottom_z, top_z = self._container_interior_bounds()
        return (top_z - bottom_z) * self.height_map_color_norm_container_ratio

    def _height_map_raycast_start_z(self, top_z):
        raycast_margin = 0.1
        placed_top_z = top_z
        for placed in self.placed_objects:
            _, aabb_max = self.p.getAABB(placed["body_id"])
            placed_top_z = max(placed_top_z, float(aabb_max[2]))
        return placed_top_z + raycast_margin

    def _scan_container_height_map(self):
        x_min, _, y_min, _, bottom_z, top_z = self._container_interior_bounds()
        l_cells, w_cells, _ = self.container_size
        x_axis = x_min + (np.arange(l_cells) + 0.5) * self.xy_resolution
        y_axis = y_min + (np.arange(w_cells) + 0.5) * self.xy_resolution
        ray_start_z = self._height_map_raycast_start_z(top_z)
        ray_from, ray_to = [], []
        for x in x_axis:
            for y in y_axis:
                ray_from.append([float(x), float(y), ray_start_z])
                ray_to.append([float(x), float(y), bottom_z - 0.1])
        hits = []
        for s in range(0, len(ray_from), 1024):
            hits.extend(self.p.rayTestBatch(ray_from[s:s + 1024], ray_to[s:s + 1024]))
        height_map = np.zeros((len(x_axis), len(y_axis)), dtype=float)
        for idx, hit in enumerate(hits):
            ix = idx // len(y_axis)
            iy = idx % len(y_axis)
            if hit[0] != -1:
                height_map[ix, iy] = max(0.0, float(hit[3][2]) - bottom_z)
        return height_map, x_axis, y_axis

    def _local_max_height(self, height_map, x_axis, y_axis, x, y, half_x, half_y):
        ix0 = max(0, int(np.searchsorted(x_axis, x - half_x, side="left")))
        ix1 = min(len(x_axis) - 1, int(np.searchsorted(x_axis, x + half_x, side="right") - 1))
        iy0 = max(0, int(np.searchsorted(y_axis, y - half_y, side="left")))
        iy1 = min(len(y_axis) - 1, int(np.searchsorted(y_axis, y + half_y, side="right") - 1))
        if ix0 > ix1 or iy0 > iy1:
            return 0.0
        return float(np.max(height_map[ix0:ix1 + 1, iy0:iy1 + 1]))

    # ---------------- 3D Scene Rendering ----------------

    def _compute_camera_matrices(self):
        # Compute the camera view/projection matrices. The view depends only on container and buffer bounds,
        # These matrices are constant within one sample and are cached at reset time.
        l_cells, w_cells, _ = self.container_size
        container_x_min = self.target_origin[0]
        container_x_max = self.target_origin[0] + self.wall_width * 2 + l_cells * self.xy_resolution
        container_y_min = self.target_origin[1]
        container_y_max = self.target_origin[1] + self.wall_width * 2 + w_cells * self.xy_resolution
        scene_x_min, scene_x_max = container_x_min, container_x_max
        scene_y_min, scene_y_max = container_y_min, container_y_max
        # Keep both the buffer area and container in view.
        if hasattr(self, "gray_buffer_bounds_xy"):
            bx0, bx1, by0, by1 = self.gray_buffer_bounds_xy
            scene_x_min = min(scene_x_min, bx0)
            scene_x_max = max(scene_x_max, bx1)
            scene_y_min = min(scene_y_min, by0)
            scene_y_max = max(scene_y_max, by1)
        scene_center = np.array([(scene_x_min + scene_x_max) / 2.0, (scene_y_min + scene_y_max) / 2.0], dtype=float)
        scene_span = max(scene_x_max - scene_x_min, scene_y_max - scene_y_min)
        camera_distance = max(2.05, scene_span * 1.04)
        far_val = max(10.0, camera_distance + scene_span * 2.0 + 2.0)
        target_center = np.array([scene_center[0], scene_center[1], 0.04], dtype=float)
        scene_z_max = max(0.65, self.container_size[2] * self.z_resolution + 0.05)
        bbox_points = np.array(
            [[x, y, z] for x in (scene_x_min, scene_x_max) for y in (scene_y_min, scene_y_max) for z in (0.0, scene_z_max)],
            dtype=float,
        )
        render_width = self.render_width * self.render_aa_scale
        render_height = self.render_height * self.render_aa_scale
        projection_matrix = self.p.computeProjectionMatrixFOV(
            fov=55, aspect=float(render_width) / float(render_height), nearVal=0.01, farVal=far_val
        )
            # Iteratively nudge the target so the bounding-box projection is centered.
        for _ in range(3):
            view_matrix = self.p.computeViewMatrixFromYawPitchRoll(target_center, camera_distance, yaw=40, pitch=-43, roll=0, upAxisIndex=2)
            projected = project_bbox_ndc_center(bbox_points, view_matrix, projection_matrix)
            if projected is None:
                break
            error = -projected
            if float(np.linalg.norm(error)) < 0.015:
                break
            eps = max(0.02, scene_span * 0.015)
            jacobian = np.zeros((2, 2), dtype=float)
            ok = True
            for axis in range(2):
                shifted = target_center.copy()
                shifted[axis] += eps
                sview = self.p.computeViewMatrixFromYawPitchRoll(shifted, camera_distance, yaw=40, pitch=-43, roll=0, upAxisIndex=2)
                scenter = project_bbox_ndc_center(bbox_points, sview, projection_matrix)
                if scenter is None:
                    ok = False
                    break
                jacobian[:, axis] = (scenter - projected) / eps
            if not ok:
                break
            try:
                delta = np.linalg.lstsq(jacobian, error, rcond=None)[0]
            except Exception:
                break
            max_delta = scene_span * 0.18
            dn = float(np.linalg.norm(delta))
            if dn > max_delta:
                delta = delta / dn * max_delta
            target_center[:2] += delta
        view_matrix = self.p.computeViewMatrixFromYawPitchRoll(target_center, camera_distance, yaw=40, pitch=-43, roll=0, upAxisIndex=2)
        self._cached_view_matrix = view_matrix
        self._cached_projection_matrix = projection_matrix

    def capture_scene_rgb(self, width=None, height=None):
        width = int(width or self.render_width)
        height = int(height or self.render_height)
        render_width = width * self.render_aa_scale
        render_height = height * self.render_aa_scale
        # Reuse the camera matrices cached at reset time.
        view_matrix = self._cached_view_matrix
        projection_matrix = self._cached_projection_matrix
        _, _, rgb, depth_buffer, segmentation = self.p.getCameraImage(
            width=render_width, height=render_height, viewMatrix=view_matrix, projectionMatrix=projection_matrix,
            shadow=0, lightDirection=[-0.55, -0.35, -0.82], lightColor=[1.0, 0.97, 0.92], lightDistance=4.0,
            lightAmbientCoeff=0.68, lightDiffuseCoeff=0.62, lightSpecularCoeff=0.08, renderer=self.p.ER_TINY_RENDERER,
        )
        image = np.reshape(rgb, (render_height, render_width, 4))[:, :, :3].astype(np.uint8)
        bg_mask = depth_background_mask(depth_buffer, render_height, render_width)
        bg_mask |= segmentation_body_mask(segmentation, render_height, render_width, getattr(self, "ground_body_id", 0))
        if self.render_aa_scale > 1:
            image = resize_rgb_lanczos(image, height, width)
            bg_mask = resize_mask_nearest(bg_mask, height, width)
        image = enhance_rgb(image, brightness=1.28, contrast=1.10, gamma=0.78)
        return apply_ground_background(image, bg_mask, view_matrix, projection_matrix)

    def colored_height_map_image(self, scale=4, alpha=1.0):
        # Upscale and colorize the height map for saved visualizations.
        height_map, h_max = self.scan_height_map()
        denom = max(1e-8, h_max)
        mask = height_map > 0
        n = np.clip(height_map / denom, 0.0, 1.0)
        small = np.zeros((height_map.shape[0], height_map.shape[1], 4), dtype=float)
        small[:, :, 0] = np.where(mask, n, 0.0)           # R: low to high maps from 0 to 1.
        small[:, :, 2] = np.where(mask, 1.0 - n, 1.0)     # B: low to high maps from 1 to 0; empty cells stay blue.
        small[:, :, 3] = alpha
        # Nearest-neighbor upscale; each grid cell becomes scale by scale pixels.
        return np.kron(small, np.ones((scale, scale, 1)))
