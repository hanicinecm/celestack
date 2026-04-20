"""Tests for photutils-based background subtraction."""

import numpy as np
import pytest

from celestack.proxy._background import subtract_background


def test_subtract_background_zeros_foreground() -> None:
    """Masked (foreground) pixels are exactly zero in the output."""
    rng = np.random.default_rng(1)
    arr = rng.random((32, 32), dtype=np.float32).astype(np.float16) * 0.1 + 0.5
    mask = np.zeros((32, 32), dtype=np.bool_)
    mask[:8, :] = True
    out = subtract_background(arr, mask, box_size=8, filter_size=1)
    assert out.dtype == np.float16
    assert np.all(out[mask] == np.float16(0.0))


def test_subtract_background_sky_centered_near_zero() -> None:
    """After subtraction, sky pixels are roughly zero-mean."""
    rng = np.random.default_rng(2)
    sky_gradient = (
        np.linspace(0.3, 0.7, 32).astype(np.float32)[:, None].repeat(32, axis=1)
    )
    arr = (sky_gradient + rng.normal(0, 0.01, (32, 32))).astype(np.float16)
    mask = np.zeros((32, 32), dtype=np.bool_)
    out = subtract_background(arr, mask, box_size=8, filter_size=1)
    assert abs(float(np.mean(out))) < 0.05


def test_subtract_background_shape_mismatch_raises() -> None:
    """Mismatched array and mask shapes raise ValueError."""
    arr = np.zeros((16, 16), dtype=np.float16)
    mask = np.zeros((8, 16), dtype=np.bool_)
    with pytest.raises(ValueError, match="Shape mismatch"):
        subtract_background(arr, mask, box_size=4, filter_size=1)
