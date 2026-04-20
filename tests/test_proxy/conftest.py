"""Fixtures for the proxy test sub-package."""

from pathlib import Path

import numpy as np
import pytest

from celestack.frame.core import Frame
from celestack.mask.core import Mask
from tests.utils import write_rgb_tiff


@pytest.fixture()
def rgb_frame(tmp_path: Path) -> Frame:
    """64x64 uint16 RGB frame with mild gradient and a few bright peaks."""
    rng = np.random.default_rng(0)
    base = rng.integers(1000, 3000, size=(64, 64, 3), dtype=np.uint16)
    # Gradient so Background2D has something to fit
    yy = np.linspace(0, 2000, 64, dtype=np.uint16)[:, None, None]
    base = np.clip(base.astype(np.int32) + yy.astype(np.int32), 0, 65535).astype(
        np.uint16
    )
    # A couple of bright star-like peaks
    base[10, 10] = 60000
    base[40, 50] = 55000
    path = tmp_path / "rgb.tif"
    write_rgb_tiff(path, base)
    return Frame(path)


@pytest.fixture()
def sky_mask(rgb_frame: Frame) -> Mask:
    """Mask where bottom 8 rows are foreground, rest is sky."""
    h, w = rgb_frame.shape[:2]
    array = np.zeros((h, w), dtype=np.bool_)
    array[-8:, :] = True
    return Mask._from_array(array)
