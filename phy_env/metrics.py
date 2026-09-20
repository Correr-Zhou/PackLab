"""Shared final packing metrics for dataset generation and evaluation."""

from __future__ import annotations


def object_volume(obj: dict) -> float:
    size = obj["size_m"]
    return float(size[0] * size[1] * size[2])


def final_packing_metrics(env, total_object_volume: float, total_object_count: int) -> dict:
    """Return center-xyz final packing metrics for a finished episode."""
    x_min, x_max, y_min, y_max, bottom_z, top_z = env._container_interior_bounds()
    center_xyz_volume = 0.0
    center_xyz_max_top_z = None
    placed = env.get_placed_objects()
    for obj in placed:
        x, y, z = obj["position"]
        volume = object_volume(obj)
        center_xyz_inside = (
            x_min <= float(x) <= x_max
            and y_min <= float(y) <= y_max
            and bottom_z <= float(z) <= top_z
        )
        if center_xyz_inside:
            _, aabb_max = env.p.getAABB(int(obj["body_id"]))
            center_xyz_volume += volume
            center_xyz_max_top_z = max(center_xyz_max_top_z or float("-inf"), float(aabb_max[2]))
    total_volume = max(1e-12, float(total_object_volume))
    container_base_area = max(1e-12, (x_max - x_min) * (y_max - y_min))
    center_xyz_height = max(0.0, center_xyz_max_top_z - bottom_z) if center_xyz_max_top_z is not None else 0.0
    center_xyz_compactness = (
        center_xyz_volume / (container_base_area * center_xyz_height) if center_xyz_height > 1e-12 else 0.0
    )
    center_xyz_volume_ratio = center_xyz_volume / total_volume
    center_xyz_overall_score = center_xyz_volume_ratio * center_xyz_compactness
    return {
        "center_xyz_inside_volume": center_xyz_volume,
        "center_xyz_inside_volume_ratio": center_xyz_volume_ratio,
        "center_xyz_inside_occupied_height": center_xyz_height,
        "center_xyz_inside_container_compactness": center_xyz_compactness,
        "center_xyz_overall_packing_score": center_xyz_overall_score,
        "total_object_count": int(total_object_count),
        "inside_volume": center_xyz_volume,
        "total_volume": float(total_object_volume),
        "inside_volume_ratio": center_xyz_volume_ratio,
        "processed_object_count": len(placed),
        "processed_object_ratio": len(placed) / max(1, int(total_object_count)),
    }
