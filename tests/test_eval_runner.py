import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf
import pandas as pd

from eval.action_sequences import export_action_sequences
from eval.metrics import summarize_cases
from eval.report import write_eval_outputs
from eval.vllm_runner import (
    _diagnostic_image,
    evaluate_cases_with_client,
    load_eval_cases,
    retry_request,
    resolve_scoring_config_path,
    rollout_case_with_client,
)


def test_summarize_cases_reports_score_metrics():
    cases = [
        {"difficulty": "easy", "buffer_size": 3, "final_score": 1.0, "compactness_raw": 0.98, "status": "ok"},
        {"difficulty": "hard", "buffer_size": 10, "final_score": -0.2, "compactness_raw": 0.8, "status": "failed"},
    ]
    summary = summarize_cases(cases)
    assert summary["overall"]["num_cases"] == 2
    assert summary["by_difficulty"]["easy"]["num_cases"] == 1


def test_public_eval_config_uses_short_thinking_token_budget():
    cfg = OmegaConf.load("configs/eval/packlab_bench.yaml")

    assert cfg.generation.max_tokens == 512
    prompt_cfg = OmegaConf.load(cfg.prompt.config_path)
    assert prompt_cfg.allow_thinking is True


def test_public_eval_scripts_disable_thinking_by_default():
    legacy_env_name = "PACKLAB_DISABLE" + "_THINKING"
    for path in [Path("scripts/evaluate.sh"), Path("scripts/evaluate_openai.sh")]:
        text = path.read_text(encoding="utf-8")

        assert "prompt.allow_thinking=false" in text
        assert legacy_env_name not in text


def test_summarize_cases_reports_success_failure_and_timing():
    cases = [
        {
            "difficulty": "easy",
            "buffer_size": 3,
            "final_score": 1.0,
            "compactness_raw": 0.98,
            "status": "ok",
            "termination_reason": "completed",
            "invalid_action_count": 0,
            "raw_invalid_action_count": 1,
            "parse_fail_count": 0,
            "invalid_object_id_count": 0,
            "raw_out_of_bounds_x_count": 1,
            "raw_out_of_bounds_y_count": 0,
            "raw_exceed_height_count": 0,
            "total_steps": 2,
            "elapsed_s": 10.0,
            "center_xyz_inside_volume_ratio": 0.4,
            "center_xyz_inside_container_compactness": 0.2,
            "center_xyz_overall_packing_score": 0.08,
            "processed_object_ratio": 1.0,
            "steps": [
                {"raw_action": {"x": 0, "y": 0}, "reason": "out_of_bounds_x"},
                {"raw_action": {"x": 1, "y": 0}},
            ],
        },
        {
            "difficulty": "hard",
            "buffer_size": 10,
            "final_score": -0.2,
            "compactness_raw": 0.8,
            "status": "ok",
            "termination_reason": "max_invalid_actions",
            "invalid_action_count": 2,
            "raw_invalid_action_count": 2,
            "parse_fail_count": 1,
            "invalid_object_id_count": 1,
            "raw_out_of_bounds_x_count": 0,
            "raw_out_of_bounds_y_count": 1,
            "raw_exceed_height_count": 1,
            "total_steps": 3,
            "elapsed_s": 20.0,
            "center_xyz_inside_volume_ratio": 0.1,
            "center_xyz_inside_container_compactness": 0.05,
            "center_xyz_overall_packing_score": 0.005,
            "processed_object_ratio": 0.5,
            "steps": [
                {"raw_action": {"x": 0, "y": 0}, "reason": "out_of_bounds_y"},
                {"raw_action": {"x": 0, "y": 0}, "reason": "exceed_height"},
                {"raw_action": {"x": 2, "y": 0}},
            ],
        },
        {
            "difficulty": "hard",
            "buffer_size": 10,
            "status": "api_error",
            "error": "server failed",
            "elapsed_s": 5.0,
        },
    ]
    summary = summarize_cases(cases)
    assert summary["overall"]["num_cases"] == 3
    assert summary["overall"]["num_ok"] == 2
    assert summary["overall"]["pack_success_rate"] == 1 / 3
    assert summary["overall"]["mean_invalid_action_count"] == 2 / 3
    assert summary["overall"]["avg_seconds_per_case"] == 35.0 / 3
    assert summary["overall"]["raw_invalid_action_ratio"] == pytest.approx(7 / 12)
    assert summary["overall"]["parse_fail_ratio"] == pytest.approx(1 / 6)
    assert summary["overall"]["invalid_object_id_ratio"] == pytest.approx(1 / 6)
    assert summary["overall"]["raw_out_of_bounds_action_ratio"] == pytest.approx(7 / 12)
    assert summary["overall"]["raw_00_ratio"] == pytest.approx(3 / 5)
    assert summary["overall"]["first_step_raw_00_ratio"] == 1.0
    assert summary["overall"]["max_consecutive_raw_00"] == 2
    assert summary["overall"]["mean_center_xyz_inside_volume_ratio"] == pytest.approx(1 / 6)
    assert summary["overall"]["mean_center_xyz_inside_container_compactness"] == pytest.approx(1 / 12)
    assert summary["overall"]["mean_center_xyz_overall_packing_score"] == pytest.approx(0.085 / 3)
    assert summary["overall"]["termination_reason_counts"]["completed"] == 1
    assert summary["overall"]["termination_reason_counts"]["max_invalid_actions"] == 1
    assert summary["overall"]["status_counts"]["api_error"] == 1


def test_retry_request_continues_until_success():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient error")
        return "ok"

    assert retry_request(fn, max_retries=3, retry_sleep_s=0) == "ok"
    assert calls["n"] == 3


def test_retry_request_raises_after_retries():
    with pytest.raises(RuntimeError):
        retry_request(lambda: (_ for _ in ()).throw(RuntimeError("bad")), max_retries=1, retry_sleep_s=0)


def test_load_eval_cases_limits_cases_per_difficulty(tmp_path):
    rows = []
    for difficulty in ["easy", "medium", "hard"]:
        for idx in range(3):
            rows.append(
                {
                    "extra_info": {
                        "episode_id": f"{difficulty}_{idx}",
                        "difficulty": difficulty,
                        "interaction_kwargs": {"buffer_size": idx + 1},
                    }
                }
            )
    path = tmp_path / "eval.parquet"
    pd.DataFrame(rows).to_parquet(path)

    cases = load_eval_cases(path, max_cases_per_difficulty=2)

    assert [case["sample_id"] for case in cases] == [
        "easy_0",
        "easy_1",
        "medium_0",
        "medium_1",
        "hard_0",
        "hard_1",
    ]


def test_load_eval_cases_reads_multiple_paths(tmp_path):
    paths = []
    for difficulty in ["easy", "hard"]:
        path = tmp_path / difficulty / "test.parquet"
        path.parent.mkdir()
        pd.DataFrame(
            [
                {
                    "extra_info": {
                        "episode_id": f"{difficulty}_0",
                        "difficulty": difficulty,
                        "interaction_kwargs": {"buffer_size": 3 if difficulty == "easy" else 10},
                    }
                }
            ]
        ).to_parquet(path)
        paths.append(str(path))

    cases = load_eval_cases(paths)

    assert [case["sample_id"] for case in cases] == ["easy_0", "hard_0"]


def test_evaluate_cases_records_error_and_continues():
    def client(case):
        if case["sample_id"] == "bad":
            raise RuntimeError("api failed")
        return {"sample_id": case["sample_id"], "status": "ok", "final_score": 1.0, "compactness_raw": 1.0}

    result = evaluate_cases_with_client(
        [{"sample_id": "ok", "difficulty": "easy"}, {"sample_id": "bad", "difficulty": "hard"}],
        client,
        max_retries=1,
        retry_sleep_s=0,
        fail_fast=False,
    )

    assert len(result) == 2
    assert result[0]["status"] == "ok"
    assert result[1]["status"] == "api_error"
    assert "api failed" in result[1]["error"]


def test_evaluate_cases_records_elapsed_seconds():
    result = evaluate_cases_with_client(
        [{"sample_id": "ok", "difficulty": "easy"}],
        lambda case: {"sample_id": case["sample_id"], "status": "ok", "final_score": 1.0, "compactness_raw": 1.0},
        max_retries=0,
        retry_sleep_s=0,
        fail_fast=False,
    )

    assert result[0]["elapsed_s"] >= 0.0


def test_evaluate_cases_fail_fast_raises():
    with pytest.raises(RuntimeError):
        evaluate_cases_with_client(
            [{"sample_id": "bad", "difficulty": "hard"}],
            lambda case: (_ for _ in ()).throw(RuntimeError("api failed")),
            max_retries=0,
            retry_sleep_s=0,
            fail_fast=True,
        )


def test_write_eval_outputs(tmp_path):
    cases = [{"sample_id": "a", "difficulty": "easy", "buffer_size": 3, "final_score": 1.0, "compactness_raw": 1.0}]
    write_eval_outputs(tmp_path, cases)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "cases.jsonl").exists()
    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "visualizations").is_dir()


def test_export_action_sequences_keeps_replay_ready_actions(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "action_sequences.jsonl"
    case = {
        "sample_id": "0",
        "difficulty": "easy",
        "buffer_size": 3,
        "status": "ok",
        "termination_reason": "completed",
        "extra_info": {
            "interaction_kwargs": {
                "sample_id": "easy_000000",
                "container_size_cm": [55.0, 50.0, 35.0],
                "buffer_size": 3,
                "object_sequence": [
                    {"internal_id": "obj_00000", "type_id": 0, "size_cm": [10.0, 20.0, 5.0]},
                    {"internal_id": "obj_00001", "type_id": 1, "size_cm": [15.0, 10.0, 5.0]},
                ],
                "seed": 610000,
            }
        },
        "steps": [
            {
                "raw_output": "{\"object_id\":0,\"x\":1,\"y\":2,\"rotation\":0}",
                "raw_action": {"object_id": 0, "x": 1, "y": 2, "rotation": 0},
                "action": {"object_id": 0, "x": 1, "y": 2, "rotation": 0},
                "valid": True,
                "reason": None,
                "parse_event": "strict_json",
            },
            {
                "raw_output": "{\"object_id\":0,\"x\":3,\"y\":4,\"rotation\":1}",
                "raw_action": {"object_id": 0, "x": 3, "y": 4, "rotation": 1},
                "action": {"object_id": 0, "x": 3, "y": 4, "rotation": 1},
                "valid": True,
                "reason": None,
                "parse_event": "strict_json",
            },
        ],
    }
    cases_path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = export_action_sequences(cases_path, output_path)

    assert summary == {"cases": 1, "steps": 2, "output_path": str(output_path)}
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["case_id"] == "easy_000000"
    assert rows[0]["eval_sample_id"] == "0"
    assert rows[0]["container_size_cm"] == [55.0, 50.0, 35.0]
    assert rows[0]["object_sequence"][1]["internal_id"] == "obj_00001"
    assert rows[0]["actions"] == [
        {
            "step_index": 0,
            "visible_objects": [
                {
                    "object_id": 0,
                    "buffer_id": 0,
                    "window_index": 1,
                    "seq_index": 1,
                    "internal_id": "obj_00001",
                    "type_id": 1,
                    "size_cm": [15.0, 10.0, 5.0],
                },
                {
                    "object_id": 1,
                    "buffer_id": 1,
                    "window_index": 0,
                    "seq_index": 0,
                    "internal_id": "obj_00000",
                    "type_id": 0,
                    "size_cm": [10.0, 20.0, 5.0],
                },
            ],
            "selected_object": {
                "object_id": 0,
                "buffer_id": 0,
                "window_index": 1,
                "seq_index": 1,
                "internal_id": "obj_00001",
                "type_id": 1,
                "size_cm": [15.0, 10.0, 5.0],
            },
            "raw_selected_object": {
                "object_id": 0,
                "buffer_id": 0,
                "window_index": 1,
                "seq_index": 1,
                "internal_id": "obj_00001",
                "type_id": 1,
                "size_cm": [15.0, 10.0, 5.0],
            },
            "raw_output": "{\"object_id\":0,\"x\":1,\"y\":2,\"rotation\":0}",
            "raw_action": {"object_id": 0, "x": 1, "y": 2, "rotation": 0},
            "action": {"object_id": 0, "x": 1, "y": 2, "rotation": 0},
            "valid": True,
            "reason": None,
            "parse_event": "strict_json",
            "forced_outside": False,
            "forced_outside_reason": None,
        },
        {
            "step_index": 1,
            "visible_objects": [
                {
                    "object_id": 0,
                    "buffer_id": 0,
                    "window_index": 0,
                    "seq_index": 0,
                    "internal_id": "obj_00000",
                    "type_id": 0,
                    "size_cm": [10.0, 20.0, 5.0],
                },
            ],
            "selected_object": {
                "object_id": 0,
                "buffer_id": 0,
                "window_index": 0,
                "seq_index": 0,
                "internal_id": "obj_00000",
                "type_id": 0,
                "size_cm": [10.0, 20.0, 5.0],
            },
            "raw_selected_object": {
                "object_id": 0,
                "buffer_id": 0,
                "window_index": 0,
                "seq_index": 0,
                "internal_id": "obj_00000",
                "type_id": 0,
                "size_cm": [10.0, 20.0, 5.0],
            },
            "raw_output": "{\"object_id\":0,\"x\":3,\"y\":4,\"rotation\":1}",
            "raw_action": {"object_id": 0, "x": 3, "y": 4, "rotation": 1},
            "action": {"object_id": 0, "x": 3, "y": 4, "rotation": 1},
            "valid": True,
            "reason": None,
            "parse_event": "strict_json",
            "forced_outside": False,
            "forced_outside_reason": None,
        },
    ]


def test_write_eval_outputs_summary_includes_rich_metrics(tmp_path):
    cases = [
        {
            "sample_id": "a",
            "difficulty": "easy",
            "buffer_size": 3,
            "status": "ok",
            "termination_reason": "completed",
            "final_score": 1.0,
            "compactness_raw": 1.0,
            "invalid_action_count": 0,
            "elapsed_s": 2.0,
        }
    ]
    write_eval_outputs(tmp_path, cases)
    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "## Quality" in text
    assert "## Action Reliability" in text
    assert "## Collapse Diagnostics" in text
    assert "## Completion" in text
    assert "Pack success rate" in text
    assert "Mean center-xyz overall packing score" in text
    assert "Mean center-xyz inside occupied height" not in text
    assert "Raw invalid action ratio" in text
    assert "Raw (0,0) ratio" in text
    assert "Average seconds per case" in text
    assert "completed" in text
    assert "Obj*Comp" not in text
    assert "Vol*Comp" not in text


def test_write_eval_outputs_summary_uses_final_metric_report_fields(tmp_path):
    cases = [
        {
            "sample_id": "a",
            "difficulty": "easy",
            "buffer_size": 3,
            "status": "ok",
            "termination_reason": "completed",
            "final_score": 1.0,
            "compactness_raw": 0.9,
            "center_xyz_inside_volume_ratio": 0.6,
            "center_xyz_inside_container_compactness": 0.3,
            "center_xyz_overall_packing_score": 0.18,
            "inside_volume_ratio": 0.8,
            "processed_object_ratio": 1.0,
            "elapsed_s": 2.0,
        }
    ]

    write_eval_outputs(tmp_path, cases)

    text = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "Mean center-xyz overall packing score" in text
    assert "Object Ratio" not in text
    assert "Mean compactness raw" not in text
    assert "Mean final score" not in text
    assert "Mean inside object ratio" not in text
    assert "Mean inside volume ratio" not in text
    assert "Final Score" not in text
    assert "Compactness" not in text


def test_rollout_case_with_client_runs_episode_and_reports_breakdown():
    calls = []

    def client(messages):
        calls.append(messages)
        return '{"object_id":0,"x":0,"y":0,"rotation":0}'

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": "configs/prompts/packing_sft.yaml"},
            "episode_control": {"max_invalid_actions": 3, "max_turns_buffer": 1, "hard_max_turns": 4},
        }
    )
    case = {
        "sample_id": "tiny_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [{"internal_id": "obj_0", "type_id": 0, "size_cm": [5.0, 5.0, 5.0]}],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert result["sample_id"] == "tiny_case"
    assert result["status"] == "ok"
    assert result["total_steps"] >= 1
    assert result["termination_reason"] == "completed"
    assert result["final_score"] == result["score_breakdown"]["final_score"]
    assert "compactness_raw" in result
    assert len(result["steps"]) == result["total_steps"]
    assert len(result["final_placed_objects"]) == 1
    placed = result["final_placed_objects"][0]
    assert placed["placed_index"] == 0
    assert placed["type_id"] == 0
    assert placed["size_m"] == [0.05, 0.05, 0.05]
    assert len(placed["position"]) == 3
    assert len(placed["aabb_min"]) == 3
    assert len(placed["aabb_max"]) == 3
    assert placed["center_xyz_inside"] is True
    assert isinstance(placed["aabb_xy_inside"], bool)
    assert isinstance(placed["aabb_z_inside"], bool)
    assert isinstance(placed["aabb_inside_container"], bool)
    assert placed["volume"] == pytest.approx(0.05 * 0.05 * 0.05)
    assert calls
    assert calls[0][0]["role"] == "system"
    assert calls[0][1]["role"] == "user"
    assert calls[0][1]["content"][0]["type"] == "image_url"


def test_rollout_case_with_client_diagnostics_default_keeps_full_history_without_payload_log():
    calls = []

    def client(messages):
        calls.append(messages)
        return '{"object_id":0,"x":0,"y":0,"rotation":0}'

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": "configs/prompts/packing_sft.yaml"},
            "episode_control": {"max_invalid_actions": 2, "max_turns_buffer": 1, "hard_max_turns": 4},
            "diagnostics": {"image_mode": "normal", "history_mode": "full", "log_prompt_payload": False},
        }
    )
    case = {
        "sample_id": "diagnostics_default_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [
                    {"internal_id": "obj_0", "type_id": 0, "size_cm": [5.0, 5.0, 5.0]},
                    {"internal_id": "obj_1", "type_id": 1, "size_cm": [5.0, 5.0, 5.0]},
                ],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert len(calls) == 2
    assert [message["role"] for message in calls[0]] == ["system", "user", "assistant"]
    assert [message["role"] for message in calls[1]] == ["system", "user", "assistant", "user", "assistant"]
    assert "prompt_payload" not in result["steps"][0]
    assert result["steps"][0]["diagnostics"] is None


def test_rollout_case_with_client_diagnostics_current_only_sends_only_current_turn():
    calls = []

    def client(messages):
        calls.append(messages)
        return '{"object_id":0,"x":0,"y":0,"rotation":0}'

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": "configs/prompts/packing_sft.yaml"},
            "episode_control": {"max_invalid_actions": 2, "max_turns_buffer": 1, "hard_max_turns": 4},
            "diagnostics": {"history_mode": "current_only", "log_prompt_payload": True},
        }
    )
    case = {
        "sample_id": "diagnostics_current_only_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [
                    {"internal_id": "obj_0", "type_id": 0, "size_cm": [5.0, 5.0, 5.0]},
                    {"internal_id": "obj_1", "type_id": 1, "size_cm": [5.0, 5.0, 5.0]},
                ],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert len(calls) == 2
    assert [message["role"] for message in calls[0]] == ["system", "user", "assistant"]
    assert [message["role"] for message in calls[1]] == ["system", "user", "assistant"]
    assert result["steps"][1]["diagnostics"]["message_count"] == 2
    assert result["steps"][1]["diagnostics"]["assistant_turn_count"] == 0
    assert result["steps"][1]["diagnostics"]["user_turn_count"] == 1


def test_rollout_case_with_client_diagnostics_no_assistant_removes_action_history():
    calls = []

    def client(messages):
        calls.append(messages)
        return '{"object_id":0,"x":0,"y":0,"rotation":0}'

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": "configs/prompts/packing_sft.yaml"},
            "episode_control": {"max_invalid_actions": 2, "max_turns_buffer": 1, "hard_max_turns": 4},
            "diagnostics": {"history_mode": "no_assistant", "log_prompt_payload": True},
        }
    )
    case = {
        "sample_id": "diagnostics_no_assistant_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [
                    {"internal_id": "obj_0", "type_id": 0, "size_cm": [5.0, 5.0, 5.0]},
                    {"internal_id": "obj_1", "type_id": 1, "size_cm": [5.0, 5.0, 5.0]},
                ],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert len(calls) == 2
    assert [message["role"] for message in calls[1]] == ["system", "user", "user", "assistant"]
    assert result["steps"][1]["diagnostics"]["message_count"] == 3
    assert result["steps"][1]["diagnostics"]["assistant_turn_count"] == 0
    assert result["steps"][1]["diagnostics"]["user_turn_count"] == 2


def test_rollout_case_with_client_diagnostics_blank_image_uses_stable_image_hash():
    def client(_messages):
        return '{"object_id":0,"x":0,"y":0,"rotation":0}'

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": "configs/prompts/packing_sft.yaml"},
            "episode_control": {"max_invalid_actions": 2, "max_turns_buffer": 1, "hard_max_turns": 4},
            "diagnostics": {"image_mode": "blank", "history_mode": "full", "log_prompt_payload": True},
        }
    )
    case = {
        "sample_id": "diagnostics_blank_image_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [
                    {"internal_id": "obj_0", "type_id": 0, "size_cm": [5.0, 5.0, 5.0]},
                    {"internal_id": "obj_1", "type_id": 1, "size_cm": [5.0, 5.0, 5.0]},
                ],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    hashes = [step["diagnostics"]["image_hash"] for step in result["steps"]]
    assert len(hashes) == 2
    assert hashes[0] == hashes[1]
    assert result["steps"][0]["diagnostics"]["image_mode"] == "blank"
    assert result["steps"][0]["diagnostics"]["history_mode"] == "full"


def test_diagnostic_image_solid_color_modes_preserve_shape_and_dtype():
    import numpy as np

    image = np.zeros((2, 3, 3), dtype=np.uint8)

    red = _diagnostic_image(image, {"image_mode": "solid_red"}, {}, 0)
    blue = _diagnostic_image(image, {"image_mode": "solid_blue"}, {}, 0)

    assert red.shape == image.shape
    assert blue.shape == image.shape
    assert red.dtype == np.uint8
    assert blue.dtype == np.uint8
    assert red.tolist() == [[[255, 0, 0]] * 3] * 2
    assert blue.tolist() == [[[0, 0, 255]] * 3] * 2


def test_rollout_case_with_client_records_parse_fail_and_fallback():
    def client(_messages):
        return "not json"

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": "configs/prompts/packing_sft.yaml"},
            "episode_control": {"max_invalid_actions": 1, "max_turns_buffer": 1, "hard_max_turns": 4},
        }
    )
    case = {
        "sample_id": "bad_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [{"internal_id": "obj_0", "type_id": 0, "size_cm": [5.0, 5.0, 5.0]}],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert result["status"] == "ok"
    assert result["invalid_action_count"] == 1
    assert result["format_events"] == ["parse_fail"]
    assert result["termination_reason"] == "max_invalid_actions"
    assert result["steps"][0]["action"] == {"object_id": -1, "x": 0, "y": 0, "rotation": 0}


def test_rollout_case_with_client_reasks_json_only_when_enabled():
    calls = []

    def client(messages):
        calls.append(messages)
        if len(calls) == 1:
            return "The object fits at the origin, so I will place object 0 at x=0, y=0."
        return '{"object_id":0,"x":0,"y":0,"rotation":0}'

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": "configs/prompts/packing_sft.yaml"},
            "parsing": {"reask_on_parse_fail": {"enabled": True}},
            "episode_control": {"max_invalid_actions": 1, "max_turns_buffer": 1, "hard_max_turns": 4},
        }
    )
    case = {
        "sample_id": "reask_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [{"internal_id": "obj_0", "type_id": 0, "size_cm": [5.0, 5.0, 5.0]}],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert len(calls) == 2
    assert result["termination_reason"] == "completed"
    assert result["invalid_action_count"] == 0
    assert result["parse_fail_count"] == 0
    assert result["format_events"] == ["strict_json"]
    assert result["steps"][0]["raw_output"] == '{"object_id":0,"x":0,"y":0,"rotation":0}'
    assert result["steps"][0]["action"] == {"object_id": 0, "x": 0, "y": 0, "rotation": 0}
    assert calls[1][-1]["role"] == "user"
    assert "Return only one JSON object" in calls[1][-1]["content"]


def test_rollout_case_with_client_can_disable_thinking_for_reask_only(tmp_path):
    calls = []

    def client(messages, request_overrides=None):
        calls.append((messages, request_overrides))
        if len(calls) == 1:
            return "I should place the object at the origin."
        return '{"object_id":0,"x":0,"y":0,"rotation":0}'

    prompt_path = tmp_path / "prompt_without_thinking_flag.yaml"
    prompt_path.write_text(
        "\n".join(
            [
                "template_id: prompt_without_thinking_flag",
                "response_format: json_only",
                "system_template: 'Return JSON.'",
                "user_template: '<image>\\nCurrent visible buffer objects:\\n{buffer_objects_json}'",
            ]
        ),
        encoding="utf-8",
    )
    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": str(prompt_path)},
            "parsing": {"reask_on_parse_fail": {"enabled": True, "disable_thinking": True}},
            "episode_control": {"max_invalid_actions": 1, "max_turns_buffer": 1, "hard_max_turns": 4},
        }
    )
    case = {
        "sample_id": "reask_disable_thinking_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [{"internal_id": "obj_0", "type_id": 0, "size_cm": [5.0, 5.0, 5.0]}],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert result["termination_reason"] == "completed"
    assert calls[0][1] is None
    assert calls[1][1] == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def test_rollout_case_with_client_enables_thinking_from_prompt_config():
    calls = []

    def client(messages, request_overrides=None):
        calls.append((messages, request_overrides))
        return '{"object_id":0,"x":0,"y":0,"rotation":0}'

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {
                "config_path": "configs/prompts/packing_sft.yaml"
            },
            "episode_control": {"max_invalid_actions": 1, "max_turns_buffer": 1, "hard_max_turns": 4},
        }
    )
    case = {
        "sample_id": "thinking_enabled_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [{"internal_id": "obj_0", "type_id": 0, "size_cm": [5.0, 5.0, 5.0]}],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert result["termination_reason"] == "completed"
    assert result["format_events"] == ["strict_json"]
    assert calls[0][0][-1] == {"role": "assistant", "content": "Pick a valid placement directly.\n</think>\n\n"}
    assert calls[0][1] == {
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
            "add_generation_prompt": False,
            "continue_final_message": True,
        },
    }


def test_rollout_case_with_client_adjusts_xy_bounds_when_enabled():
    def client(_messages):
        return '{"object_id":0,"x":99,"y":99,"rotation":0}'

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": "configs/prompts/packing_sft.yaml"},
            "episode_control": {"max_invalid_actions": 1, "max_turns_buffer": 1, "hard_max_turns": 4},
            "action_adjustment": {"xy_bounds": {"enabled": True, "mode": "random", "seed": 7}},
        }
    )
    case = {
        "sample_id": "adjust_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [{"internal_id": "obj_0", "type_id": 0, "size_cm": [5.0, 5.0, 5.0]}],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert result["termination_reason"] == "completed"
    assert result["invalid_action_count"] == 0
    assert result["raw_invalid_action_count"] == 1
    step = result["steps"][0]
    assert step["raw_action"] == {"object_id": 0, "x": 99, "y": 99, "rotation": 0}
    assert step["action"]["object_id"] == 0
    assert 0 <= step["action"]["x"] <= 6
    assert 0 <= step["action"]["y"] <= 6
    assert step["action_adjustment"]["type"] == "xy_bounds_random"
    assert {c["rotation"] for c in step["action_adjustment"]["candidates"]} == {0, 1}
    assert step["valid"] is True


def test_rollout_case_with_client_adjusts_impossible_rotation_before_step():
    def client(_messages):
        return '{"object_id":0,"x":0,"y":0,"rotation":0}'

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": "configs/prompts/packing_sft.yaml"},
            "episode_control": {"max_invalid_actions": 1, "max_turns_buffer": 1, "hard_max_turns": 4},
            "action_adjustment": {"xy_bounds": {"enabled": True, "mode": "random", "seed": 7}},
        }
    )
    case = {
        "sample_id": "rotation_adjust_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 10.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [{"internal_id": "obj_0", "type_id": 0, "size_cm": [10.0, 20.0, 5.0]}],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert result["termination_reason"] == "completed"
    assert result["invalid_action_count"] == 0
    assert result["raw_invalid_action_count"] == 1
    step = result["steps"][0]
    assert step["raw_action"] == {"object_id": 0, "x": 0, "y": 0, "rotation": 0}
    assert step["action"]["rotation"] == 1
    assert step["action_adjustment"]["type"] == "xy_bounds_random"
    assert step["action_adjustment"]["reason"] == "rotation_0_has_no_xy_fit"
    assert step["action_adjustment"]["candidates"] == [{"rotation": 1, "valid_x_range": [0, 0], "valid_y_range": [0, 0]}]
    assert step["valid"] is True


def test_rollout_case_with_client_adjusts_height_before_step():
    calls = {"n": 0}

    def client(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return '{"object_id":0,"x":0,"y":0,"rotation":0}'
        return '{"object_id":0,"x":0,"y":0,"rotation":0}'

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": "configs/prompts/packing_sft.yaml"},
            "episode_control": {"max_invalid_actions": 1, "max_turns_buffer": 1, "hard_max_turns": 4},
            "action_adjustment": {
                "xy_bounds": {"enabled": True, "mode": "random", "seed": 7},
                "height": {"enabled": True, "mode": "random_feasible", "max_attempts": 64, "seed": 7},
            },
        }
    )
    case = {
        "sample_id": "height_adjust_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [
                    {"internal_id": "obj_0", "type_id": 0, "size_cm": [10.0, 20.0, 12.0]},
                    {"internal_id": "obj_1", "type_id": 1, "size_cm": [10.0, 10.0, 12.0]},
                ],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert result["invalid_action_count"] == 0
    assert result["raw_invalid_action_count"] >= 1
    assert result["adjustment_count"] >= 1
    assert any((step.get("action_adjustment") or {}).get("type") == "height_random_feasible" for step in result["steps"])
    for step in result["steps"]:
        assert step["valid"] is True


def test_rollout_case_with_client_relaxed_places_out_of_bounds_and_reports_inside_metrics():
    def client(_messages):
        return '{"object_id":0,"x":20,"y":0,"rotation":0}'

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": "configs/prompts/packing_sft.yaml"},
            "placement": {"mode": "relaxed_physical", "forced_outside_seed": 11},
            "episode_control": {"max_invalid_actions": None, "max_turns_buffer": 1, "hard_max_turns": 4},
            "action_adjustment": {
                "xy_bounds": {"enabled": False},
                "height": {"enabled": False},
            },
        }
    )
    case = {
        "sample_id": "relaxed_outside_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [{"internal_id": "obj_0", "type_id": 0, "size_cm": [5.0, 5.0, 5.0]}],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert result["termination_reason"] == "completed"
    assert result["total_steps"] == 1
    assert result["invalid_action_count"] == 1
    assert result["raw_out_of_bounds_x_count"] == 1
    assert result["inside_volume_ratio"] == 0.0
    assert result["center_xyz_inside_volume_ratio"] == 0.0
    assert result["center_xyz_inside_container_compactness"] == 0.0
    assert result["center_xyz_overall_packing_score"] == 0.0
    assert result["inside_volume_ratio"] == result["center_xyz_inside_volume_ratio"]
    object_prefix = "_".join(["center", "xyz", "inside", "object", ""])
    assert all(not key.startswith(object_prefix) for key in result)
    assert "inside_volume_compactness" not in result
    assert result["steps"][0]["valid"] is False
    assert result["steps"][0]["placement_mode"] == "relaxed_physical"
    assert len(result["final_placed_objects"]) == 1
    placed = result["final_placed_objects"][0]
    assert placed["center_xyz_inside"] is False
    assert placed["aabb_xy_inside"] is False
    assert placed["aabb_inside_container"] is False


def test_rollout_case_with_client_can_strip_think_tags_before_parsing_action():
    def client(_messages):
        return '<think>{"object_id": 0, "x": 99, "y": 99, "rotation": 0}</think>\n{"object_id":0,"x":0,"y":0,"rotation":0}'

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": "configs/prompts/packing_sft.yaml"},
            "parsing": {"strip_think_tags": True},
            "placement": {"mode": "relaxed_physical", "forced_outside_seed": 11},
            "episode_control": {"max_invalid_actions": None, "max_turns_buffer": 1, "hard_max_turns": 4},
            "action_adjustment": {
                "xy_bounds": {"enabled": False},
                "height": {"enabled": False},
            },
        }
    )
    case = {
        "sample_id": "relaxed_think_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [{"internal_id": "obj_0", "type_id": 0, "size_cm": [5.0, 5.0, 5.0]}],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert result["termination_reason"] == "completed"
    assert result["invalid_action_count"] == 0
    assert result["steps"][0]["action"] == {"object_id": 0, "x": 0, "y": 0, "rotation": 0}
    assert result["format_events"] == ["strict_json"]


def test_rollout_case_with_client_relaxed_forces_parse_fail_outside_and_continues():
    calls = {"n": 0}

    def client(_messages):
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json"
        return '{"object_id":0,"x":0,"y":0,"rotation":0}'

    cfg = OmegaConf.create(
        {
            "env": {"config_path": "configs/phy_env/env.yaml"},
            "prompt": {"config_path": "configs/prompts/packing_sft.yaml"},
            "placement": {"mode": "relaxed_physical", "forced_outside_seed": 11},
            "episode_control": {"max_invalid_actions": None, "max_turns_buffer": 1, "hard_max_turns": 4},
            "action_adjustment": {
                "xy_bounds": {"enabled": False},
                "height": {"enabled": False},
            },
        }
    )
    case = {
        "sample_id": "relaxed_parse_fail_case",
        "difficulty": "easy",
        "extra_info": {
            "interaction_kwargs": {
                "container_size_cm": [20.0, 20.0, 20.0],
                "buffer_size": 1,
                "object_sequence": [
                    {"internal_id": "obj_0", "type_id": 0, "size_cm": [5.0, 5.0, 5.0]},
                    {"internal_id": "obj_1", "type_id": 1, "size_cm": [5.0, 5.0, 5.0]},
                ],
                "seed": 1,
            }
        },
    }

    result = rollout_case_with_client(case, cfg, client)

    assert result["termination_reason"] == "completed"
    assert result["total_steps"] == 2
    assert result["parse_fail_count"] == 1
    assert result["invalid_object_id_count"] == 0
    assert result["forced_outside_count"] == 1
    assert result["processed_object_ratio"] == 1.0
    assert result["total_object_count"] == 2
    assert result["inside_volume_ratio"] == pytest.approx(0.5)
    assert result["steps"][0]["forced_outside"] is True
    assert result["steps"][0]["valid"] is False
    assert result["steps"][1]["valid"] is True


def test_resolve_scoring_config_path_uses_eval_config_value():
    cfg = OmegaConf.create({"scoring": {"config_path": "custom/scoring.yaml"}})

    assert resolve_scoring_config_path(cfg) == "custom/scoring.yaml"


def test_resolve_scoring_config_path_defaults_to_packing_score():
    cfg = OmegaConf.create({})

    assert resolve_scoring_config_path(cfg) == "configs/phy_env/scoring/packing_score.yaml"


def test_default_eval_config_uses_relaxed_placement_without_xy_bounds_adjust():
    cfg = OmegaConf.load("configs/eval/packlab_bench.yaml")

    assert cfg.placement.mode == "relaxed_physical"
    assert cfg.episode_control.max_invalid_actions is None
    assert cfg.action_adjustment.xy_bounds.enabled is False
