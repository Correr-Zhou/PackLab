from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _normalize_object(obj: dict, *, object_id: int | None = None, window_index: int | None = None) -> dict:
    record = dict(obj)
    if object_id is not None:
        record["object_id"] = int(object_id)
        record["buffer_id"] = int(object_id)
    if window_index is not None:
        record["window_index"] = int(window_index)
    return record


class _ReplayBufferWindow:
    def __init__(self, object_sequence: list[dict], buffer_size: int, *, shuffle_visible_objects: bool, seed: int | None):
        self._sequence = [dict(obj) for obj in object_sequence]
        self._buffer_size = int(buffer_size)
        self._shuffle_visible_objects = bool(shuffle_visible_objects)
        self._rng = random.Random(seed)
        self._next_idx = 0
        self._window: list[dict] = []
        self._visible_order: list[int] = []
        self._fill()
        self._refresh_visible_order()

    def _fill(self) -> None:
        while len(self._window) < self._buffer_size and self._next_idx < len(self._sequence):
            obj = dict(self._sequence[self._next_idx])
            obj["seq_index"] = self._next_idx
            self._window.append(obj)
            self._next_idx += 1

    def _refresh_visible_order(self) -> None:
        self._visible_order = list(range(len(self._window)))
        if self._shuffle_visible_objects:
            self._rng.shuffle(self._visible_order)

    def get_visible_objects(self) -> list[dict]:
        visible = []
        for visible_id, window_idx in enumerate(self._visible_order):
            visible.append(_normalize_object(self._window[window_idx], object_id=visible_id, window_index=window_idx))
        return visible

    def advance(self, picked_buffer_id: int) -> bool:
        if picked_buffer_id < 0 or picked_buffer_id >= len(self._visible_order):
            return False
        window_idx = self._visible_order[picked_buffer_id]
        self._window.pop(window_idx)
        self._fill()
        self._refresh_visible_order()
        return True


def _case_id(case: dict) -> str:
    interaction_kwargs = case.get("extra_info", {}).get("interaction_kwargs", {}) or {}
    return str(
        interaction_kwargs.get("sample_id")
        or case.get("extra_info", {}).get("episode_id")
        or case.get("sample_id")
        or ""
    )


def _selected_object(visible_objects: list[dict], action: dict | None) -> dict | None:
    if not isinstance(action, dict):
        return None
    object_id = action.get("object_id")
    if not isinstance(object_id, int) or object_id < 0 or object_id >= len(visible_objects):
        return None
    return visible_objects[object_id]


def _should_advance_buffer(step: dict, selected_object: dict | None) -> bool:
    if selected_object is None:
        return False
    if step.get("placement_mode") == "relaxed_physical":
        return True
    return bool(step.get("valid", False))


def _action_steps(case: dict, *, shuffle_visible_objects: bool) -> list[dict]:
    interaction_kwargs = case.get("extra_info", {}).get("interaction_kwargs", {}) or {}
    replay_buffer = _ReplayBufferWindow(
        interaction_kwargs.get("object_sequence", []),
        int(interaction_kwargs.get("buffer_size", case.get("buffer_size", 0)) or 0),
        shuffle_visible_objects=shuffle_visible_objects,
        seed=interaction_kwargs.get("seed"),
    )
    actions = []
    for idx, step in enumerate(case.get("steps", []) or []):
        visible_objects = replay_buffer.get_visible_objects()
        selected_object = _selected_object(visible_objects, step.get("action"))
        raw_selected_object = _selected_object(visible_objects, step.get("raw_action"))
        actions.append(
            {
                "step_index": int(step.get("step_index", idx)),
                "visible_objects": visible_objects,
                "selected_object": selected_object,
                "raw_selected_object": raw_selected_object,
                "raw_output": step.get("raw_output"),
                "raw_action": step.get("raw_action"),
                "action": step.get("action"),
                "valid": bool(step.get("valid", False)),
                "reason": step.get("reason"),
                "parse_event": step.get("parse_event"),
                "forced_outside": bool(step.get("forced_outside", False)),
                "forced_outside_reason": step.get("forced_outside_reason"),
            }
        )
        if _should_advance_buffer(step, selected_object):
            replay_buffer.advance(int(step["action"]["object_id"]))
    return actions


def export_action_sequences(
    cases_path: str | Path,
    output_path: str | Path,
    *,
    shuffle_visible_objects: bool = True,
) -> dict:
    cases_path = Path(cases_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    case_count = 0
    step_count = 0
    with cases_path.open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            case = json.loads(line)
            interaction_kwargs = case.get("extra_info", {}).get("interaction_kwargs", {}) or {}
            actions = _action_steps(case, shuffle_visible_objects=shuffle_visible_objects)
            record = {
                "case_id": _case_id(case),
                "eval_sample_id": str(case.get("sample_id", "")),
                "difficulty": case.get("difficulty"),
                "status": case.get("status"),
                "termination_reason": case.get("termination_reason"),
                "container_size_cm": interaction_kwargs.get("container_size_cm"),
                "buffer_size": interaction_kwargs.get("buffer_size", case.get("buffer_size")),
                "seed": interaction_kwargs.get("seed"),
                "object_sequence": interaction_kwargs.get("object_sequence", []),
                "actions": actions,
            }
            dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            case_count += 1
            step_count += len(actions)

    return {"cases": case_count, "steps": step_count, "output_path": str(output_path)}


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export replay-ready action sequences from eval cases.jsonl.")
    parser.add_argument("--cases", required=True, help="Path to eval cases.jsonl.")
    parser.add_argument("--output", required=True, help="Output action_sequences.jsonl path.")
    parser.add_argument(
        "--shuffle-visible-objects",
        type=_parse_bool,
        default=True,
        help="Whether to replay the buffer id mapping with the eval env's visible-object shuffle.",
    )
    args = parser.parse_args()

    print(
        json.dumps(
            export_action_sequences(
                args.cases,
                args.output,
                shuffle_visible_objects=args.shuffle_visible_objects,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
