from __future__ import annotations

from collections import Counter, defaultdict


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _mean_key(items, key: str) -> float:
    return _mean([float(x.get(key, 0.0)) for x in items])


def _ratio_by_case(items, numerator_key: str, denominator_key: str = "total_steps") -> float:
    values = []
    for item in items:
        denominator = float(item.get(denominator_key, 0.0) or 0.0)
        if denominator <= 0:
            continue
        values.append(float(item.get(numerator_key, 0.0) or 0.0) / denominator)
    return _mean(values)


def _raw_out_of_bounds_action_ratio(items) -> float:
    values = []
    for item in items:
        denominator = float(item.get("total_steps", 0.0) or 0.0)
        if denominator <= 0:
            continue
        numerator = (
            float(item.get("raw_out_of_bounds_x_count", 0.0) or 0.0)
            + float(item.get("raw_out_of_bounds_y_count", 0.0) or 0.0)
            + float(item.get("raw_exceed_height_count", 0.0) or 0.0)
        )
        values.append(numerator / denominator)
    return _mean(values)


def _raw_xy_collapse(items) -> dict:
    total_steps = 0
    raw_00 = 0
    first_step_cases = 0
    first_step_raw_00 = 0
    max_consecutive_raw_00 = 0
    for item in items:
        run = 0
        steps = item.get("steps", []) or []
        if steps:
            first_step_cases += 1
        for step_pos, step in enumerate(steps):
            total_steps += 1
            raw_action = step.get("raw_action") or {}
            is_00 = (raw_action.get("x"), raw_action.get("y")) == (0, 0)
            if is_00:
                raw_00 += 1
                run += 1
                max_consecutive_raw_00 = max(max_consecutive_raw_00, run)
            else:
                run = 0
            if step_pos == 0 and is_00:
                first_step_raw_00 += 1
    return {
        "raw_00_ratio": raw_00 / total_steps if total_steps else 0.0,
        "first_step_raw_00_ratio": first_step_raw_00 / first_step_cases if first_step_cases else 0.0,
        "max_consecutive_raw_00": max_consecutive_raw_00,
    }


def _count_by_key(items, key: str):
    return dict(Counter(str(x.get(key, "unknown")) for x in items if x.get(key) is not None))


def _is_pack_success(case: dict) -> bool:
    return case.get("status") == "ok" and case.get("termination_reason") == "completed"


def _summarize(items):
    num_cases = len(items)
    elapsed_values = [float(x.get("elapsed_s", 0.0)) for x in items]
    collapse = _raw_xy_collapse(items)
    return {
        "num_cases": num_cases,
        "num_ok": sum(1 for x in items if x.get("status") == "ok"),
        "pack_success_rate": sum(1 for x in items if _is_pack_success(x)) / num_cases if num_cases else 0.0,
        "mean_final_score": _mean([float(x.get("final_score", 0.0)) for x in items]),
        "mean_compactness_raw": _mean([float(x.get("compactness_raw", 0.0)) for x in items]),
        "mean_invalid_action_count": _mean([float(x.get("invalid_action_count", 0.0)) for x in items]),
        "mean_raw_invalid_action_count": _mean_key(items, "raw_invalid_action_count"),
        "mean_parse_fail_count": _mean_key(items, "parse_fail_count"),
        "mean_invalid_object_id_count": _mean_key(items, "invalid_object_id_count"),
        "raw_invalid_action_ratio": _ratio_by_case(items, "raw_invalid_action_count"),
        "parse_fail_ratio": _ratio_by_case(items, "parse_fail_count"),
        "invalid_object_id_ratio": _ratio_by_case(items, "invalid_object_id_count"),
        "raw_out_of_bounds_action_ratio": _raw_out_of_bounds_action_ratio(items),
        **collapse,
        "mean_forced_outside_count": _mean_key(items, "forced_outside_count"),
        "mean_raw_out_of_bounds_x_count": _mean_key(items, "raw_out_of_bounds_x_count"),
        "mean_raw_out_of_bounds_y_count": _mean_key(items, "raw_out_of_bounds_y_count"),
        "mean_raw_exceed_height_count": _mean_key(items, "raw_exceed_height_count"),
        "mean_center_xyz_inside_volume_ratio": _mean_key(items, "center_xyz_inside_volume_ratio"),
        "mean_center_xyz_inside_container_compactness": _mean_key(items, "center_xyz_inside_container_compactness"),
        "mean_center_xyz_overall_packing_score": _mean_key(items, "center_xyz_overall_packing_score"),
        "mean_inside_volume_ratio": _mean_key(items, "inside_volume_ratio"),
        "mean_processed_object_ratio": _mean_key(items, "processed_object_ratio"),
        "avg_seconds_per_case": sum(elapsed_values) / num_cases if num_cases else 0.0,
        "termination_reason_counts": _count_by_key(items, "termination_reason"),
        "status_counts": _count_by_key(items, "status"),
    }


def summarize_cases(cases: list[dict]) -> dict:
    by_difficulty = defaultdict(list)
    by_buffer = defaultdict(list)
    for case in cases:
        by_difficulty[str(case.get("difficulty"))].append(case)
        by_buffer[str(case.get("buffer_size"))].append(case)
    return {
        "overall": _summarize(cases),
        "by_difficulty": {k: _summarize(v) for k, v in by_difficulty.items()},
        "by_buffer_size": {k: _summarize(v) for k, v in by_buffer.items()},
    }
