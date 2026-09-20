"""Unit tests for PolicyLLMJson parsing."""

from omegaconf import OmegaConf

from phy_env.components.policy import PolicyLLMJson


def _policy():
    cfg = OmegaConf.create({"type": "llm_json", "pose_mode": "z_rotation_2"})
    return PolicyLLMJson(cfg)


def _policy_strip_think():
    cfg = OmegaConf.create({"type": "llm_json", "pose_mode": "z_rotation_2", "strip_think_tags": True})
    return PolicyLLMJson(cfg)


def test_parse_plain_json():
    action, err = _policy().parse('{"object_id": 1, "x": 10, "y": 20, "rotation": 1}')
    assert err is None
    assert action == {"object_id": 1, "x": 10, "y": 20, "rotation": 1}


def test_parse_json_in_codeblock():
    # LLM-style output wrapped with think tags and a json code fence.
    text = '<think>place it</think>\n```json\n{"object_id": 0, "x": 5, "y": 5, "rotation": 0}\n```'
    action, err = _policy().parse(text)
    assert err is None
    assert action["object_id"] == 0


def test_parse_fails_when_think_block_contains_json_without_strip_switch():
    text = '<think>{"object_id": 2, "x": 99, "y": 99, "rotation": 0}</think>\n{"object_id": 0, "x": 5, "y": 6, "rotation": 1}'
    action, info = _policy().parse_with_info(text)
    assert action is None
    assert info["parse_ok"] is False
    assert info["think_tags_stripped"] is False


def test_parse_can_strip_think_block_before_extracting_final_json():
    text = '<think>{"object_id": 2, "x": 99, "y": 99, "rotation": 0}</think>\n{"object_id": 0, "x": 5, "y": 6, "rotation": 1}'
    action, info = _policy_strip_think().parse_with_info(text)
    assert action == {"object_id": 0, "x": 5, "y": 6, "rotation": 1}
    assert info["parse_ok"] is True
    assert info["strict_json"] is True
    assert info["think_tags_stripped"] is True


def test_parse_can_strip_implicit_qwen_thinking_prefix():
    text = (
        '{"object_id": 2, "x": 99, "y": 99, "rotation": 0}\n'
        "</think>\n\n"
        '{"object_id": 0, "x": 5, "y": 6, "rotation": 1}'
    )
    action, info = _policy_strip_think().parse_with_info(text)
    assert action == {"object_id": 0, "x": 5, "y": 6, "rotation": 1}
    assert info["parse_ok"] is True
    assert info["strict_json"] is True
    assert info["think_tags_stripped"] is True


def test_parse_extra_text_is_scored_after_think_blocks_are_removed():
    text = (
        '<think>try several placements before answering</think>\n'
        'final answer: {"object_id": 0, "x": 5, "y": 6, "rotation": 1}'
    )
    action, info = _policy_strip_think().parse_with_info(text)
    assert action == {"object_id": 0, "x": 5, "y": 6, "rotation": 1}
    assert info["parse_ok"] is True
    assert info["strict_json"] is False
    assert info["think_tags_stripped"] is True


def test_parse_failure():
    action, err = _policy().parse("no json here")
    assert action is None
    assert err == "json_parse_failed"


def test_invalid_rotation():
    action, err = _policy().parse('{"object_id": 0, "x": 1, "y": 1, "rotation": 3}')
    assert action is None
    assert err == "invalid_rotation"


def test_missing_field():
    action, err = _policy().parse('{"object_id": 0, "x": 1}')
    assert action is None
    assert err == "missing_or_invalid_field"


def test_parse_with_info_strict_json():
    action, info = _policy().parse_with_info('{"object_id": 1, "x": 2, "y": 3, "rotation": 0}')
    assert action == {"object_id": 1, "x": 2, "y": 3, "rotation": 0}
    assert info["parse_ok"] is True
    assert info["strict_json"] is True
    assert info["error"] is None


def test_parse_with_info_extra_text():
    action, info = _policy().parse_with_info('place here {"object_id": 1, "x": 2, "y": 3, "rotation": 0}')
    assert action["object_id"] == 1
    assert info["parse_ok"] is True
    assert info["strict_json"] is False
    assert info["error"] is None


def test_parse_with_info_parse_fail():
    action, info = _policy().parse_with_info("no json")
    assert action is None
    assert info["parse_ok"] is False
    assert info["strict_json"] is False
    assert info["error"] == "json_parse_failed"
