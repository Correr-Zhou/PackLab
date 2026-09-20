from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EpisodeControlConfig:
    max_invalid_actions: int | None = 3
    max_turns_buffer: int = 3
    hard_max_turns: int = 40


@dataclass
class EpisodeControlResult:
    done: bool
    termination_reason: str | None


class EpisodeControl:
    def __init__(self, cfg: EpisodeControlConfig, object_count: int):
        self.cfg = cfg
        self.object_count = int(object_count)
        self.total_steps = 0
        self.total_invalid_actions = 0
        self.max_turns = min(self.object_count + int(cfg.max_turns_buffer), int(cfg.hard_max_turns))

    def update(self, valid: bool, buffer_exhausted: bool) -> EpisodeControlResult:
        self.total_steps += 1
        if not valid:
            self.total_invalid_actions += 1
        if buffer_exhausted:
            return EpisodeControlResult(done=True, termination_reason="completed")
        if self.cfg.max_invalid_actions is not None and self.total_invalid_actions >= self.cfg.max_invalid_actions:
            return EpisodeControlResult(done=True, termination_reason="max_invalid_actions")
        if self.total_steps >= self.max_turns:
            return EpisodeControlResult(done=True, termination_reason="max_turns_exceeded")
        return EpisodeControlResult(done=False, termination_reason=None)
