"""Fixtures for the mask_builder test sub-package."""

from pathlib import Path

import numpy as np
import pytest

from celestack.frame.core import Frame
from celestack.mask_builder.core import MaskBuilder
from tests.utils import write_gray_tiff, write_rgb_tiff


@pytest.fixture()
def rgb_frame(tmp_path: Path) -> Frame:
    """32x24 uint16 RGB TIFF with two visually distinct halves."""
    array = np.zeros((24, 32, 3), dtype=np.uint16)
    array[:, :16, 0] = 2000
    array[:, :16, 1] = 3000
    array[:, :16, 2] = 60000
    array[:, 16:, 0] = 40000
    array[:, 16:, 1] = 30000
    array[:, 16:, 2] = 5000
    path = tmp_path / "rgb.tif"
    write_rgb_tiff(path, array)
    return Frame(path)


@pytest.fixture()
def gray_mask_path(tmp_path: Path) -> Path:
    """32x24 uint16 grayscale TIFF with a known nonzero region."""
    array = np.zeros((24, 32), dtype=np.uint16)
    array[5:15, 8:24] = 50000
    path = tmp_path / "gray_mask.tif"
    write_gray_tiff(path, array)
    return path


@pytest.fixture()
def builder(rgb_frame: Frame) -> MaskBuilder:
    """MaskBuilder with clusters computed (K=2)."""
    mb = MaskBuilder(rgb_frame)
    mb.compute_clusters(2)
    return mb
