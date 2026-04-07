"""Array processing helpers for frame transformations."""

from __future__ import annotations

import numpy as np


def scaled_preview_uint8(array: np.ndarray) -> np.ndarray:
    """Scale arbitrary image data into an 8-bit preview array."""
    if array.dtype == np.bool_:
        return array.astype(np.uint8) * 255
    if array.dtype == np.uint8:
        return array
    data = array.astype(np.float64, copy=False)
    if data.size == 0:
        return np.zeros_like(array, dtype=np.uint8)
    max_value = np.max(data)
    min_value = np.min(data)
    if max_value <= min_value:
        return np.zeros(array.shape, dtype=np.uint8)
    normalized = (data - min_value) / (max_value - min_value)
    return np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)


def convert_bit_depth(
    array: np.ndarray, source_bit_depth: int, target_bit_depth: int
) -> np.ndarray:
    """Convert an array to the requested output bit depth."""
    if target_bit_depth not in {8, 16}:
        msg = f"Unsupported target bit depth: {target_bit_depth}"
        raise ValueError(msg)

    max_target = float((1 << target_bit_depth) - 1)
    out_dtype = np.uint8 if target_bit_depth == 8 else np.uint16

    if array.dtype == np.bool_:
        return (array.astype(np.float64) * max_target).astype(out_dtype)

    if np.issubdtype(array.dtype, np.integer):
        max_source = float((1 << max(source_bit_depth, 1)) - 1)
        scaled = array.astype(np.float64) * (max_target / max_source)
        return np.clip(np.rint(scaled), 0, max_target).astype(out_dtype)

    values = array.astype(np.float64)
    max_value = np.max(values) if values.size > 0 else 0.0
    if max_value <= 0:
        return np.zeros(array.shape, dtype=out_dtype)
    scaled = values * (max_target / max_value)
    return np.clip(np.rint(scaled), 0, max_target).astype(out_dtype)


def downscale_by_block_average(array: np.ndarray, factor: int) -> np.ndarray:
    """Downscale an image by averaging non-overlapping pixel blocks."""
    if factor == 1:
        return array.copy()
    if factor < 1:
        msg = "downscale_factor must be >= 1"
        raise ValueError(msg)

    height, width = array.shape[:2]
    new_height = max(1, height // factor)
    new_width = max(1, width // factor)
    trim_height = new_height * factor
    trim_width = new_width * factor
    trimmed = array[:trim_height, :trim_width]

    if array.ndim == 2:
        reshaped = trimmed.reshape(new_height, factor, new_width, factor)
        return reshaped.mean(axis=(1, 3))

    channels = array.shape[2]
    reshaped = trimmed.reshape(new_height, factor, new_width, factor, channels)
    return reshaped.mean(axis=(1, 3))


def to_grayscale(array: np.ndarray) -> np.ndarray:
    """Convert an RGB array to luminance grayscale."""
    if array.ndim != 3:
        return array
    rgb = array[..., :3].astype(np.float64, copy=False)
    return np.tensordot(
        rgb, np.array([0.2989, 0.5870, 0.1140], dtype=np.float64), axes=([2], [0])
    )


def interpolate_bad_pixels(array: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked pixels with the mean of their valid 8-connected neighbors.

    Args:
        array: Image array (2D or 3D). Modified in place at masked positions.
        mask: Boolean 2D mask; True where a pixel is bad.

    Returns:
        The array with masked pixels replaced by neighbor means.
    """
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    float_array = array.astype(np.float64)
    valid_mask = ~mask  # True where pixels are good

    total = np.zeros_like(float_array)
    count = np.zeros(mask.shape, dtype=np.float64)

    for dy, dx in shifts:
        neighbour = np.roll(np.roll(float_array, dy, axis=0), dx, axis=1)
        valid = np.roll(np.roll(valid_mask, dy, axis=0), dx, axis=1)
        if array.ndim == 3:
            total[mask] += np.where(valid[..., np.newaxis], neighbour, 0)[mask]
        else:
            total[mask] += np.where(valid, neighbour, 0)[mask]
        count[mask] += valid[mask]

    limits = np.iinfo(array.dtype)
    if array.ndim == 3:
        count_3d = count[..., np.newaxis]
        interpolated = np.where(
            count_3d[mask] > 0,
            np.clip(
                np.rint(total[mask] / np.where(count_3d[mask] > 0, count_3d[mask], 1)),
                limits.min,
                limits.max,
            ),
            0,
        ).astype(array.dtype)
    else:
        interpolated = np.where(
            count[mask] > 0,
            np.clip(
                np.rint(total[mask] / np.where(count[mask] > 0, count[mask], 1)),
                limits.min,
                limits.max,
            ),
            0,
        ).astype(array.dtype)

    result = array.copy()
    result[mask] = interpolated
    return result


def subtract_arrays(
    left: np.ndarray,
    right: np.ndarray,
    *,
    correct_saturated: bool = False,
) -> np.ndarray:
    """Subtract arrays while preserving the left dtype.

    Args:
        left: Minuend array.
        right: Subtrahend array.
        correct_saturated: If True and the arrays are unsigned integers, pixels
            where *right* equals the dtype maximum are replaced with the mean of
            their valid 8-connected neighbors in the result. Useful when *right*
            is a master dark with blown hot pixels that are also saturated on the
            light frame, which would otherwise produce spurious black spots.

    Returns:
        Result array with the same dtype as *left*.

    Raises:
        ValueError: If the dtype is unsupported.
    """
    if np.issubdtype(left.dtype, np.unsignedinteger):
        result = left.astype(np.int32) - right.astype(np.int32)
        limits = np.iinfo(left.dtype)
        clipped = np.clip(result, limits.min, limits.max).astype(left.dtype)
        if correct_saturated:
            mask = right == limits.max
            if mask.ndim == 3:
                mask = mask.any(axis=2)
            if mask.any():
                clipped = interpolate_bad_pixels(clipped, mask)
        return clipped

    msg = f"Unsupported dtype for subtraction: {left.dtype}"
    raise ValueError(msg)
