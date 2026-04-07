"""Tests for frame image processing operations."""

import numpy as np
import pytest

from celestack.frame._image_ops import (
    convert_bit_depth,
    downscale_by_block_average,
    scaled_preview_uint8,
    subtract_arrays,
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
        convert_bit_depth(arr, 8, 32)


def test_convert_bit_depth_bool_to_uint8() -> None:
    """Boolean array converts to 0/255 uint8."""
    arr = np.array([[True, False], [False, True]])
    result = convert_bit_depth(arr, 1, 8)
    assert result.dtype == np.uint8
    assert result[0, 0] == 255
    assert result[0, 1] == 0


# --- subtract_arrays ---


def test_subtract_arrays_unsigned_clamps_underflow() -> None:
    """Unsigned subtraction saturates at zero instead of wrapping."""
    left = np.array([[20, 5], [100, 0]], dtype=np.uint16)
    right = np.array([[3, 9], [20, 1]], dtype=np.uint16)

    result = subtract_arrays(left, right)

    assert result.dtype == np.uint16
    np.testing.assert_array_equal(
        result,
        np.array([[17, 0], [80, 0]], dtype=np.uint16),
    )


def test_subtract_arrays_bool_raises() -> None:
    """Boolean arrays are not supported for subtraction."""
    left = np.array([[True, False]])
    right = np.array([[False, True]])

    with pytest.raises(ValueError, match="Unsupported dtype for subtraction"):
        subtract_arrays(left, right)


def test_subtract_arrays_correct_saturated_interpolates_center() -> None:
    """Blown dark pixels are replaced by 8-neighbor mean, not clamped to zero."""
    max_val = np.iinfo(np.uint16).max
    # 3x3 light: uniform value of 1000
    light = np.full((3, 3), 1000, dtype=np.uint16)
    # 3x3 dark: center pixel is blown, rest is 100
    dark = np.full((3, 3), 100, dtype=np.uint16)
    dark[1, 1] = max_val

    result = subtract_arrays(light, dark, correct_saturated=True)

    # Non-saturated pixels subtract normally
    assert result[0, 0] == 900
    # Center pixel: result of normal subtraction of neighbors (900) averaged
    assert result[1, 1] == 900


def test_subtract_arrays_correct_saturated_default_off() -> None:
    """Without correct_saturated, blown dark pixels produce zero."""
    max_val = np.iinfo(np.uint16).max
    light = np.full((3, 3), 1000, dtype=np.uint16)
    dark = np.full((3, 3), 100, dtype=np.uint16)
    dark[1, 1] = max_val

    result = subtract_arrays(light, dark)

    assert result[1, 1] == 0


def test_subtract_arrays_correct_saturated_rgb() -> None:
    """Saturated-pixel correction works on 3-channel arrays."""
    max_val = np.iinfo(np.uint16).max
    light = np.full((3, 3, 3), 1000, dtype=np.uint16)
    dark = np.full((3, 3, 3), 100, dtype=np.uint16)
    dark[1, 1, :] = max_val

    result = subtract_arrays(light, dark, correct_saturated=True)

    # Neighbors subtract normally
    np.testing.assert_array_equal(result[0, 0], [900, 900, 900])
    # Center interpolated from neighbors (all 900)
    np.testing.assert_array_equal(result[1, 1], [900, 900, 900])


def test_subtract_arrays_correct_saturated_single_channel_triggers() -> None:
    """A pixel is corrected when any one RGB channel is saturated, not just all."""
    max_val = np.iinfo(np.uint16).max
    light = np.full((3, 3, 3), 1000, dtype=np.uint16)
    dark = np.full((3, 3, 3), 100, dtype=np.uint16)
    dark[1, 1, 0] = max_val  # only the red channel is blown

    result = subtract_arrays(light, dark, correct_saturated=True)

    # The whole pixel is interpolated from neighbors (all 900)
    np.testing.assert_array_equal(result[1, 1], [900, 900, 900])


def test_subtract_arrays_correct_saturated_corner_no_wraparound() -> None:
    """Corner saturated pixels use only in-bounds neighbors, no edge wraparound."""
    max_val = np.iinfo(np.uint16).max
    # 4x4 light: uniform 1000; dark: uniform 100 except top-left corner is blown
    light = np.full((4, 4), 1000, dtype=np.uint16)
    dark = np.full((4, 4), 100, dtype=np.uint16)
    dark[0, 0] = max_val

    result = subtract_arrays(light, dark, correct_saturated=True)

    # Corner pixel's only valid neighbors are (0,1), (1,0), (1,1) — all give 900
    assert result[0, 0] == 900
    # Pixels on the opposite edge must be unaffected (no wraparound)
    assert result[3, 3] == 900
    assert result[0, 3] == 900
    assert result[3, 0] == 900


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
