"""Episode visualization recorder for PackLab evaluation."""

from pathlib import Path

import numpy as np

from phy_env.render_utils import (
    append_held_frame,
    concat_images_vertically,
    concat_step_images,
    save_rgb_image,
    save_rgba_image,
    save_video,
    to_rgb_uint8,
)

RECORD_FPS = 10
RECORD_BUFFER_HOLD_FRAMES = 10
RECORD_FRAME_INTERVAL = 48


class EpisodeRecorder:
    def __init__(self, env, out_dir, save_video=True):
        self.env = env
        self.out_dir = Path(out_dir)
        self.save_video_flag = bool(save_video)
        self.frames = []
        self.summaries = []
        self.step_idx = 0
        self._buffer_frame = None
        self._subdirs = {
            "buffer": self.out_dir / "before_placement",
            "place": self.out_dir / "after_placement",
            "height": self.out_dir / "height_map",
            "summary": self.out_dir / "summaries",
        }
        for path in self._subdirs.values():
            path.mkdir(parents=True, exist_ok=True)

    def begin(self):
        init_frame = self.env.capture_scene_rgb()
        save_rgb_image(self._subdirs["place"] / "after_placement_step_000.png", init_frame)
        if self.save_video_flag:
            self.frames.append(init_frame)
            self.env.record_callback = self.frames.append
            self.env.record_interval = RECORD_FRAME_INTERVAL

    def before_step(self):
        self._buffer_frame = self.env.capture_scene_rgb()
        save_rgb_image(self._subdirs["buffer"] / f"before_placement_step_{self.step_idx + 1:03d}.png", self._buffer_frame)
        if self.save_video_flag:
            append_held_frame(self.frames, self._buffer_frame, RECORD_BUFFER_HOLD_FRAMES)

    def after_step(self):
        self.step_idx += 1
        after_frame = self.env.capture_scene_rgb()
        save_rgb_image(self._subdirs["place"] / f"after_placement_step_{self.step_idx:03d}.png", after_frame)
        if self.save_video_flag:
            append_held_frame(self.frames, after_frame, max(2, RECORD_BUFFER_HOLD_FRAMES // 2))
        height_rgba = self.env.colored_height_map_image()
        save_rgba_image(self._subdirs["height"] / f"height_map_step_{self.step_idx:03d}.png", height_rgba)
        height_rgb = to_rgb_uint8(np.clip(height_rgba * 255.0, 0, 255).astype(np.uint8))
        summary = concat_step_images(self._buffer_frame, after_frame, height_rgb, self.step_idx)
        save_rgb_image(self._subdirs["summary"] / f"summary_step_{self.step_idx:03d}.png", summary)
        self.summaries.append(summary)

    def finish(self):
        self.env.record_callback = None
        if self.save_video_flag and self.frames:
            save_video(self.out_dir / "placement_process.mp4", self.frames, RECORD_FPS)
        if self.summaries:
            save_rgb_image(self._subdirs["summary"] / "summary_all_steps.png", concat_images_vertically(self.summaries))
