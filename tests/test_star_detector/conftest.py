"""Shared fixtures for star detector tests."""

from __future__ import annotations

import numpy as np
import pytest

from celestack.mask.core import Mask
from celestack.proxy.frame import ProxyFrame
from celestack.proxy.mask import ProxyMask


def _make_star_image(
    height: int = 128,
    width: int = 128,
    star_positions: list[tuple[int, int]] | None = None,
    star_amplitude: float = 0.6,
    star_fwhm: float = 3.0,
    bg_std: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Create a synthetic background-subtracted float16 image with Gaussian stars.

    The array is centered near zero, with Gaussian stars rising above the
    noise floor — the shape a :class:`ProxyFrame` produces after
    background subtraction.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        star_positions: List of ``(row, col)`` positions. If ``None``, a
            default grid of 8 stars is used.
        star_amplitude: Peak amplitude above zero for each star.
        star_fwhm: FWHM of the Gaussian PSF in proxy pixels.
        bg_std: Standard deviation of the zero-mean background noise.
        seed: Random seed.

    Returns:
        2D float16 array with Gaussian stars and zero-mean noise.
    """
    rng = np.random.default_rng(seed)
    image = rng.normal(0.0, bg_std, (height, width)).astype(np.float64)

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
        gauss = star_amplitude * np.exp(
            -((yy - r) ** 2 + (xx - c) ** 2) / (2 * sigma**2)
        )
        image += gauss

    return image.astype(np.float16)


def _make_proxy_frame(array: np.ndarray, downscale_factor: int = 2) -> ProxyFrame:
    """Wrap a float16 array into a detached ProxyFrame."""
    instance = ProxyFrame.__new__(ProxyFrame)
    instance._path = None
    instance._array = array
    instance._downscale_factor = int(downscale_factor)
    return instance


def _make_proxy_mask(array: np.ndarray, downscale_factor: int = 2) -> ProxyMask:
    """Wrap a boolean array into a detached ProxyMask."""
    instance = ProxyMask.__new__(ProxyMask)
    instance._path = None
    instance._array = array.astype(np.bool_, copy=False)
    instance._downscale_factor = int(downscale_factor)
    return instance


@pytest.fixture()
def star_image_array() -> np.ndarray:
    """128x128 float16 background-subtracted image with 8 Gaussian stars."""
    return _make_star_image()


@pytest.fixture()
def proxy_frame(star_image_array: np.ndarray) -> ProxyFrame:
    """ProxyFrame backing a 128x128 synthetic starfield (downscale_factor=2)."""
    return _make_proxy_frame(star_image_array, downscale_factor=2)


@pytest.fixture()
def sky_mask_top_half() -> ProxyMask:
    """128x128 proxy mask: bottom half foreground, top half sky."""
    array = np.zeros((128, 128), dtype=np.bool_)
    array[64:, :] = True
    return _make_proxy_mask(array, downscale_factor=2)


@pytest.fixture()
def all_sky_mask() -> ProxyMask:
    """128x128 proxy mask: all sky (no foreground)."""
    array = np.zeros((128, 128), dtype=np.bool_)
    return _make_proxy_mask(array, downscale_factor=2)


@pytest.fixture()
def all_foreground_mask() -> ProxyMask:
    """128x128 proxy mask: all foreground."""
    array = np.ones((128, 128), dtype=np.bool_)
    return _make_proxy_mask(array, downscale_factor=2)


@pytest.fixture()
def shape_mismatch_mask() -> ProxyMask:
    """Proxy mask with a different shape than the 128x128 fixture frames."""
    array = np.zeros((64, 64), dtype=np.bool_)
    return _make_proxy_mask(array, downscale_factor=2)


@pytest.fixture()
def wrong_downscale_mask() -> ProxyMask:
    """Proxy mask that matches shape but not downscale_factor."""
    array = np.zeros((128, 128), dtype=np.bool_)
    return _make_proxy_mask(array, downscale_factor=4)


# The full-resolution Mask fixture is retained for any test that builds a
# ProxyMask via ProxyMask.from_mask(...).
@pytest.fixture()
def fullres_mask() -> Mask:
    """Full-resolution Mask covering the bottom half of a 256x256 image."""
    array = np.zeros((256, 256), dtype=np.bool_)
    array[128:, :] = True
    return Mask._from_array(array)
