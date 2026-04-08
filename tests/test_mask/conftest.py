"""Fixtures for the mask test sub-package."""

from pathlib import Path

import numpy as np
import pytest

from celestack.frame.core import Frame
from celestack.mask._mask_builder import MaskBuilder
from celestack.mask.core import Mask
from tests.utils import write_gray_tiff, write_rgb_tiff


@pytest.fixture()
def rgb_frame(tmp_path: Path) -> Frame:
    """32x24 uint16 RGB TIFF with two visually distinct halves."""
    array = np.zeros((24, 32, 3), dtype=np.uint16)
    # Left half: bright sky-like (high blue, low red)
    array[:, :16, 0] = 2000
    array[:, :16, 1] = 3000
    array[:, :16, 2] = 60000
    # Right half: dark terrain-like (high red/green, low blue)
    array[:, 16:, 0] = 40000
    array[:, 16:, 1] = 30000
    array[:, 16:, 2] = 5000
    path = tmp_path / "rgb.tif"
    write_rgb_tiff(path, array)
    return Frame(path)


@pytest.fixture()
def gray_mask_path(tmp_path: Path) -> Path:
    """32x24 uint16 grayscale TIFF with a known nonzero region (rows 5-14, cols 8-23)."""
    array = np.zeros((24, 32), dtype=np.uint16)
    array[5:15, 8:24] = 50000
    path = tmp_path / "gray_mask.tif"
    write_gray_tiff(path, array)
    return path


@pytest.fixture()
def rgb_mask_path(tmp_path: Path) -> Path:
    """32x24 uint8 RGB TIFF with top half nonzero (value 200 per channel)."""
    array = np.zeros((24, 32, 3), dtype=np.uint8)
    array[0:12, :, :] = 200
    path = tmp_path / "rgb_mask.tif"
    write_rgb_tiff(path, array)
    return path


@pytest.fixture()
def bool_values_mask_path(tmp_path: Path) -> Path:
    """32x24 uint8 TIFF with values 0 and 1 (not 0 and 255)."""
    array = np.zeros((24, 32), dtype=np.uint8)
    array[4:20, 4:28] = 1
    path = tmp_path / "bool_mask.tif"
    write_gray_tiff(path, array)
    return path


@pytest.fixture()
def gray_mask(gray_mask_path: Path) -> Mask:
    """Mask loaded from the gray_mask_path fixture."""
    return Mask(gray_mask_path)


@pytest.fixture()
def builder(rgb_frame: Frame) -> MaskBuilder:
    """MaskBuilder with clusters computed (K=2) on a 2x downscaled proxy.

    The source frame is 32x24, so the proxy is 16x12.
    """
    mb = MaskBuilder(rgb_frame, proxy_downscale_factor=2)
    mb.compute_clusters(2)
    return mb
