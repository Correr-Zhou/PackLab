"""Evaluation report writers."""

from __future__ import annotations

import json
from pathlib import Path

from eval.metrics import summarize_cases


def _jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def write_eval_outputs(output_dir: str | Path, cases: list[dict]) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    metrics = summarize_cases(cases)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "cases.jsonl").open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(json.dumps(_jsonable(case), ensure_ascii=False) + "\n")
    (output_dir / "summary.md").write_text(_summary_markdown(metrics), encoding="utf-8")
    return metrics


def _summary_markdown(metrics: dict) -> str:
    overall = metrics.get("overall", {})
    lines = [
        "# Evaluation Summary",
        "",
        "## Quality",
        "",
        f"- Mean center-xyz overall packing score: {overall.get('mean_center_xyz_overall_packing_score', 0.0):.6f}",
        f"- Mean center-xyz inside volume ratio: {overall.get('mean_center_xyz_inside_volume_ratio', 0.0):.6f}",
        f"- Mean center-xyz inside-container compactness: {overall.get('mean_center_xyz_inside_container_compactness', 0.0):.6f}",
        "",
        "## Action Reliability",
        "",
        f"- Raw invalid action ratio: {overall.get('raw_invalid_action_ratio', 0.0):.6f}",
        f"- Parse fail ratio: {overall.get('parse_fail_ratio', 0.0):.6f}",
        f"- Invalid object id ratio: {overall.get('invalid_object_id_ratio', 0.0):.6f}",
        f"- Raw out-of-bounds action ratio: {overall.get('raw_out_of_bounds_action_ratio', 0.0):.6f}",
        f"- Mean invalid action count: {overall.get('mean_invalid_action_count', 0.0):.6f}",
        f"- Mean raw invalid action count: {overall.get('mean_raw_invalid_action_count', 0.0):.6f}",
        f"- Mean parse fail count: {overall.get('mean_parse_fail_count', 0.0):.6f}",
        f"- Mean invalid object id count: {overall.get('mean_invalid_object_id_count', 0.0):.6f}",
        f"- Mean raw out-of-bounds x count: {overall.get('mean_raw_out_of_bounds_x_count', 0.0):.6f}",
        f"- Mean raw out-of-bounds y count: {overall.get('mean_raw_out_of_bounds_y_count', 0.0):.6f}",
        f"- Mean raw exceed height count: {overall.get('mean_raw_exceed_height_count', 0.0):.6f}",
        "",
        "## Collapse Diagnostics",
        "",
        f"- Raw (0,0) ratio: {overall.get('raw_00_ratio', 0.0):.6f}",
        f"- First-step raw (0,0) ratio: {overall.get('first_step_raw_00_ratio', 0.0):.6f}",
        f"- Max consecutive raw (0,0): {overall.get('max_consecutive_raw_00', 0)}",
        "",
        "## Completion",
        "",
        f"- Cases: {overall.get('num_cases', 0)}",
        f"- OK cases: {overall.get('num_ok', 0)}",
        f"- Pack success rate: {overall.get('pack_success_rate', 0.0):.6f}",
        f"- Mean forced outside count: {overall.get('mean_forced_outside_count', 0.0):.6f}",
        f"- Mean processed object ratio: {overall.get('mean_processed_object_ratio', 0.0):.6f}",
        f"- Average seconds per case: {overall.get('avg_seconds_per_case', 0.0):.6f}",
        "",
        "## Status Counts",
        "",
    ]
    lines.extend(_count_lines(overall.get("status_counts", {})))
    lines.extend(["", "## Termination Reasons", ""])
    lines.extend(_count_lines(overall.get("termination_reason_counts", {})))
    lines.extend(["", "## By Difficulty", ""])
    lines.extend(_breakdown_table(metrics.get("by_difficulty", {})))
    lines.extend(["", "## By Buffer Size", ""])
    lines.extend(_breakdown_table(metrics.get("by_buffer_size", {})))
    lines.append("")
    return "\n".join(lines)


def _count_lines(counts: dict) -> list[str]:
    if not counts:
        return ["- none: 0"]
    return [f"- {key}: {counts[key]}" for key in sorted(counts)]


def _breakdown_table(groups: dict) -> list[str]:
    if not groups:
        return ["No cases."]
    lines = [
        "| Group | Cases | OK | Pack Success | Invalid Actions | Raw Invalid | Center XYZ Vol | Center XYZ Comp | XYZ Overall | Processed Obj | Sec/Case |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(groups):
        item = groups[key]
        lines.append(
            "| "
            f"{key} | "
            f"{item.get('num_cases', 0)} | "
            f"{item.get('num_ok', 0)} | "
            f"{item.get('pack_success_rate', 0.0):.6f} | "
            f"{item.get('mean_invalid_action_count', 0.0):.6f} | "
            f"{item.get('mean_raw_invalid_action_count', 0.0):.6f} | "
            f"{item.get('mean_center_xyz_inside_volume_ratio', 0.0):.6f} | "
            f"{item.get('mean_center_xyz_inside_container_compactness', 0.0):.6f} | "
            f"{item.get('mean_center_xyz_overall_packing_score', 0.0):.6f} | "
            f"{item.get('mean_processed_object_ratio', 0.0):.6f} | "
            f"{item.get('avg_seconds_per_case', 0.0):.6f} |"
        )
    return lines
