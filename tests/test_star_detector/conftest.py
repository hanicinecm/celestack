"""Shared fixtures for star detector tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from celestack.frame._metadata import CELESTACK_KEY
from celestack.frame.core import Frame
from celestack.mask.core import Mask
from tests.utils import write_gray_tiff


def _make_star_image(
    height: int = 128,
    width: int = 128,
    star_positions: list[tuple[int, int]] | None = None,
    star_flux: float = 800.0,
    star_fwhm: float = 3.0,
    bg_mean: float = 100.0,
    bg_std: float = 10.0,
    seed: int = 42,
) -> np.ndarray:
    """Create a synthetic grayscale image with Gaussian stars on a noisy background.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        star_positions: List of (row, col) positions for stars. If None,
            a default grid is used.
        star_flux: Peak amplitude above background for each star.
        star_fwhm: FWHM of the Gaussian PSF.
        bg_mean: Mean background value.
        bg_std: Std of Gaussian background noise.
        seed: Random seed.

    Returns:
        2D uint16 array with stars and noise.
    """
    rng = np.random.default_rng(seed)
    image = rng.normal(bg_mean, bg_std, (height, width)).astype(np.float64)

    if star_positions is None:
        star_positions = [
            (20, 20),
            (20, 60),
            (20, 100),
            (60, 40),
            (60, 80),
            (100, 20),
            (100, 60),
            (100, 100),
        ]

    sigma = star_fwhm / 2.3548
    yy, xx = np.mgrid[0:height, 0:width]
    for r, c in star_positions:
        gauss = star_flux * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * sigma**2))
        image += gauss

    return np.clip(image, 0, 65535).astype(np.uint16)


@pytest.fixture()
def star_image_array() -> np.ndarray:
    """128x128 uint16 grayscale image with 8 Gaussian stars."""
    return _make_star_image()


@pytest.fixture()
def proxy_frame(tmp_path: Path, star_image_array: np.ndarray) -> Frame:
    """Proxy frame (downscale_factor=2) with Gaussian stars."""
    import tifffile

    path = tmp_path / "proxy.tif"
    tifffile.imwrite(
        path,
        data=star_image_array,
        photometric="minisblack",
        metadata={CELESTACK_KEY: {"downscale_factor": 2}},
    )
    return Frame(path)


@pytest.fixture()
def full_res_frame(tmp_path: Path, star_image_array: np.ndarray) -> Frame:
    """Full-res frame (downscale_factor=1) with Gaussian stars."""
    path = tmp_path / "full_res.tif"
    write_gray_tiff(path, star_image_array)
    return Frame(path)


@pytest.fixture()
def sky_mask_top_half(tmp_path: Path) -> Mask:
    """128x128 mask: bottom half is foreground (True), top half is sky (False)."""
    array = np.zeros((128, 128), dtype=np.uint8)
    array[64:, :] = 255  # bottom half is foreground
    path = tmp_path / "mask.tif"
    write_gray_tiff(path, array)
    return Mask(path)


@pytest.fixture()
def sky_mask_proxy(tmp_path: Path) -> Mask:
    """128x128 proxy mask (downscale_factor=2): bottom half foreground."""
    import tifffile

    array = np.zeros((128, 128), dtype=np.uint8)
    array[64:, :] = 255
    path = tmp_path / "mask_proxy.tif"
    tifffile.imwrite(
        path,
        data=array,
        photometric="minisblack",
        metadata={CELESTACK_KEY: {"downscale_factor": 2}},
    )
    return Mask(path)


@pytest.fixture()
def all_sky_mask(tmp_path: Path) -> Mask:
    """128x128 mask: all sky (no foreground)."""
    array = np.zeros((128, 128), dtype=np.uint8)
    path = tmp_path / "all_sky.tif"
    write_gray_tiff(path, array)
    return Mask(path)


@pytest.fixture()
def all_foreground_mask(tmp_path: Path) -> Mask:
    """128x128 mask: all foreground."""
    array = np.full((128, 128), 255, dtype=np.uint8)
    path = tmp_path / "all_fg.tif"
    write_gray_tiff(path, array)
    return Mask(path)
