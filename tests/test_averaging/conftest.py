"""Fixtures for the averaging test sub-package."""

from pathlib import Path

import numpy as np
import pytest
from tests.utils import write_gray_tiff, write_rgb_tiff

from celestack.frame import Frame


@pytest.fixture()
def gray_frames(tmp_path: Path) -> list[Frame]:
    """Three 32x24 grayscale uint16 TIFFs with known values."""
    arrays = [np.full((24, 32), fill_value=v, dtype=np.uint16) for v in [100, 200, 300]]
    frames = []
    for i, arr in enumerate(arrays):
        p = write_gray_tiff(tmp_path / f"frame_{i}.tif", arr)
        frames.append(Frame(p))
    return frames


@pytest.fixture()
def rgb_frames(tmp_path: Path) -> list[Frame]:
    """Three 32x24 RGB uint8 TIFFs with known values."""
    arrays = [
        np.full((24, 32, 3), fill_value=v, dtype=np.uint8) for v in [50, 100, 150]
    ]
    frames = []
    for i, arr in enumerate(arrays):
        p = write_rgb_tiff(tmp_path / f"rgb_{i}.tif", arr)
        frames.append(Frame(p))
    return frames
