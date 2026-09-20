"""Observation component that renders the container height map as a VLM image."""

import numpy as np

from phy_env.registry import register_component


@register_component("observation", "height_map_image")
class ObservationHeightMapImage:
    def __init__(self, cfg):
        self.cfg = cfg
        self.output_size = tuple(cfg.get("output_size", [224, 224]))
        self.color_mode = cfg.get("color_mode", "blue_to_red")

    def render(self, env):
        # Colorize the height map from blue to red, then resize to output_size.
        height_map, h_max = env.scan_height_map()
        colored = self._colorize(height_map, h_max)
        return self._resize(colored)

    def _colorize(self, height_map, h_max):
        # Low cells are blue, high cells are red, and empty cells stay blue.
        h, w = height_map.shape
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:, :, 2] = 255  # Default blue.
        denom = max(1e-8, h_max)
        mask = height_map > 0
        norm = np.clip(height_map / denom, 0.0, 1.0)
        img[:, :, 0] = np.where(mask, (norm * 255).astype(np.uint8), 0)
        img[:, :, 2] = np.where(mask, ((1.0 - norm) * 255).astype(np.uint8), 255)
        return img

    def _resize(self, img):
        # Nearest-neighbor resizing preserves grid boundaries.
        from PIL import Image

        target_w, target_h = self.output_size[1], self.output_size[0]
        pil = Image.fromarray(img)
        pil = pil.resize((target_w, target_h), Image.NEAREST)
        return np.asarray(pil, dtype=np.uint8)
