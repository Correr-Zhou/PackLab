"""Unit tests for ObservationHeightMapImage with a lightweight fake environment."""

import numpy as np
from omegaconf import OmegaConf

from phy_env.components.observation import ObservationHeightMapImage


class _FakeEnv:
    def __init__(self, height_map, h_max):
        self._hm = height_map
        self._h_max = h_max

    def scan_height_map(self):
        return self._hm, self._h_max


def _obs():
    cfg = OmegaConf.create({"type": "height_map_image", "output_size": [224, 224], "color_mode": "blue_to_red"})
    return ObservationHeightMapImage(cfg)


def test_output_shape_and_dtype():
    # Output should be a 224x224x3 uint8 image.
    hm = np.zeros((56, 56), dtype=float)
    img = _obs().render(_FakeEnv(hm, 1.0))
    assert img.shape == (224, 224, 3)
    assert img.dtype == np.uint8


def test_empty_cells_are_blue():
    # An empty container should be fully blue.
    hm = np.zeros((56, 56), dtype=float)
    img = _obs().render(_FakeEnv(hm, 1.0))
    assert img[:, :, 2].min() == 255
    assert img[:, :, 0].max() == 0


def test_high_cells_are_red():
    # Cells near h_max should be red.
    hm = np.zeros((56, 56), dtype=float)
    hm[10:20, 10:20] = 1.0
    img = _obs().render(_FakeEnv(hm, 1.0))
    # At least one pixel should reach a red-channel value of 255.
    assert img[:, :, 0].max() == 255
