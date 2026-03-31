"""Tests for frame image processing operations."""

import numpy as np
import pytest

from celestack.frame._image_ops import (
    convert_bit_depth,
    downscale_by_block_average,
    scaled_preview_uint8,
    to_grayscale,
)


# --- downscale_by_block_average ---


def test_downscale_factor_one_returns_copy() -> None:
    """Factor=1 returns a copy of the input array."""
    arr = np.arange(12, dtype=np.uint16).reshape(3, 4)
    result = downscale_by_block_average(arr, 1)
    np.testing.assert_array_equal(result, arr)
    assert result is not arr


def test_downscale_factor_below_one_raises() -> None:
    """Factor < 1 raises ValueError."""
    arr = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="downscale_factor must be >= 1"):
        downscale_by_block_average(arr, 0)


def test_downscale_trims_non_divisible() -> None:
    """Non-divisible dimensions are trimmed before averaging."""
    # 5x7 with factor=2 → trims to 4x6 → output 2x3
    arr = np.ones((5, 7), dtype=np.float64)
    result = downscale_by_block_average(arr, 2)
    assert result.shape == (2, 3)


def test_downscale_3d_preserves_channels() -> None:
    """Block averaging on 3D array preserves the channel dimension."""
    arr = np.ones((8, 8, 3), dtype=np.uint16)
    result = downscale_by_block_average(arr, 2)
    assert result.shape == (4, 4, 3)


# --- to_grayscale ---


def test_to_grayscale_passthrough_2d() -> None:
    """2D input passes through unchanged."""
    arr = np.arange(12, dtype=np.uint8).reshape(3, 4)
    result = to_grayscale(arr)
    np.testing.assert_array_equal(result, arr)


def test_to_grayscale_luminance_formula() -> None:
    """RGB conversion uses the ITU-R BT.601 luminance formula."""
    # Pure red pixel
    arr = np.array([[[255, 0, 0]]], dtype=np.uint8)
    result = to_grayscale(arr)
    expected = 0.2989 * 255
    np.testing.assert_allclose(result[0, 0], expected, rtol=1e-4)


# --- convert_bit_depth ---


def test_convert_bit_depth_unsupported_raises() -> None:
    """Unsupported target bit depth raises ValueError."""
    arr = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported target bit depth"):
        convert_bit_depth(arr, 8, 12)


def test_convert_bit_depth_bool_to_uint8() -> None:
    """Boolean array converts to 0/255 uint8."""
    arr = np.array([[True, False], [False, True]])
    result = convert_bit_depth(arr, 1, 8)
    assert result.dtype == np.uint8
    assert result[0, 0] == 255
    assert result[0, 1] == 0


def test_convert_bit_depth_bool_to_float32() -> None:
    """Boolean array converts to float32."""
    arr = np.array([[True, False]])
    result = convert_bit_depth(arr, 1, 32)
    assert result.dtype == np.float32
    assert result[0, 0] == 1.0
    assert result[0, 1] == 0.0


def test_convert_bit_depth_float_to_uint8() -> None:
    """Float array normalizes to max value when converting to uint8."""
    arr = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    result = convert_bit_depth(arr, 32, 8)
    assert result.dtype == np.uint8
    assert result[0, 0] == 0
    assert result[0, 2] == 255


# --- scaled_preview_uint8 ---


def test_scaled_preview_bool() -> None:
    """Boolean array scales to 0 and 255."""
    arr = np.array([[True, False], [False, True]])
    result = scaled_preview_uint8(arr)
    assert result.dtype == np.uint8
    assert result[0, 0] == 255
    assert result[0, 1] == 0


def test_scaled_preview_constant() -> None:
    """Constant array produces all zeros."""
    arr = np.full((3, 3), 42, dtype=np.uint16)
    result = scaled_preview_uint8(arr)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result, 0)


def test_scaled_preview_empty() -> None:
    """Empty array returns empty zeros."""
    arr = np.zeros((0, 4), dtype=np.uint16)
    result = scaled_preview_uint8(arr)
    assert result.dtype == np.uint8
    assert result.shape == (0, 4)
