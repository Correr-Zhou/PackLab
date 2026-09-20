from __future__ import annotations

import json


def _compact_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def build_system_message(container_state: dict, prompt_cfg) -> dict:
    content = str(prompt_cfg["system_template"]).replace("{container_json}", _compact_json(container_state))
    return {"role": "system", "content": content}


def build_user_text(buffer_state: list[dict], prompt_cfg) -> str:
    return str(prompt_cfg["user_template"]).replace("{buffer_objects_json}", _compact_json(buffer_state))


def build_user_message(buffer_state: list[dict], prompt_cfg) -> dict:
    return {"role": "user", "content": build_user_text(buffer_state, prompt_cfg)}


def build_assistant_message(raw_output: str) -> dict:
    return {"role": "assistant", "content": str(raw_output)}
