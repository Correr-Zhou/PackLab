"""Policy component that parses LLM JSON outputs into environment actions."""

import json
import re

from phy_env.registry import register_component


@register_component("policy", "llm_json")
class PolicyLLMJson:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pose_mode = cfg.get("pose_mode", "z_rotation_2")
        self.strip_think_tags = bool(cfg.get("strip_think_tags", False))

    def parse(self, llm_output):
        """Extract a JSON action from LLM text.

        Returns (action, error). On success, action is a dict and error is None.
        On failure, action is None and error describes the parse issue.
        """
        action, info = self.parse_with_info(llm_output)
        return action, info["error"]

    def parse_with_info(self, llm_output):
        parse_text, think_tags_stripped = self._strip_think_blocks(llm_output)
        obj, strict_json = self._extract_json_with_strict(parse_text)
        if obj is None:
            return None, {
                "parse_ok": False,
                "strict_json": False,
                "think_tags_stripped": think_tags_stripped,
                "error": "json_parse_failed",
            }
        try:
            action = {
                "object_id": int(obj["object_id"]),
                "x": int(obj["x"]),
                "y": int(obj["y"]),
                "rotation": int(obj.get("rotation", 0)),
            }
        except (KeyError, ValueError, TypeError):
            return None, {
                "parse_ok": False,
                "strict_json": strict_json,
                "think_tags_stripped": think_tags_stripped,
                "error": "missing_or_invalid_field",
            }
        # z_rotation_2 only allows rotations 0 and 1.
        if self.pose_mode == "z_rotation_2" and action["rotation"] not in (0, 1):
            return None, {
                "parse_ok": False,
                "strict_json": strict_json,
                "think_tags_stripped": think_tags_stripped,
                "error": "invalid_rotation",
            }
        return action, {
            "parse_ok": True,
            "strict_json": strict_json,
            "think_tags_stripped": think_tags_stripped,
            "error": None,
        }

    def _extract_json(self, text):
        # Prefer parsing the whole text; otherwise extract the first JSON-like object.
        obj, _ = self._extract_json_with_strict(text)
        return obj

    def _extract_json_with_strict(self, text):
        if isinstance(text, dict):
            return text, True
        try:
            return json.loads(text), True
        except (json.JSONDecodeError, TypeError):
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0)), False
            except json.JSONDecodeError:
                return None, False
        return None, False

    def _strip_think_blocks(self, text):
        if not self.strip_think_tags or not isinstance(text, str):
            return text, False
        stripped = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        if stripped == text and "</think>" in text.lower():
            stripped = re.split(r"</think>", text, maxsplit=1, flags=re.IGNORECASE)[1]
        return stripped, stripped != text
