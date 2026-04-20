"""Array processing helpers for frame transformations."""

from __future__ import annotations

import numpy as np

from celestack.config import CFG


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

    Uses the mean rather than the median to preserve channel correlation in
    RGB images — per-channel median selects values from different neighbor
    pixels, producing a synthetic color that introduces a systematic bias
    visible after frame averaging.

    Operates only on the small set of masked pixel coordinates rather than
    rolling the full image array, so cost scales with the number of bad pixels,
    not with image size.

    Args:
        array: Image array (2D or 3D).
        mask: Boolean 2D mask; True where a pixel is bad.

    Returns:
        A copy of *array* with masked pixels replaced by neighbor means.
        Pixels with no valid neighbors are set to zero.
    """
    result = array.copy()
    ys, xs = np.where(mask)
    if ys.size == 0:
        return result

    h, w = array.shape[:2]
    limits = np.iinfo(array.dtype)
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y, x in zip(ys, xs):
        neighbour_vals = [
            array[y + dy, x + dx]
            for dy, dx in shifts
            if 0 <= y + dy < h and 0 <= x + dx < w and not mask[y + dy, x + dx]
        ]
        if neighbour_vals:
            mean = np.mean(neighbour_vals, axis=0)
            result[y, x] = np.clip(np.rint(mean), limits.min, limits.max).astype(
                array.dtype
            )
        else:
            result[y, x] = 0

    return result


def build_dark_bad_pixel_mask(dark_array: np.ndarray) -> np.ndarray:
    """Identify hot pixels in a dark frame using a MAD-based outlier threshold.

    Computes per-pixel brightness (max across channels for RGB arrays), then
    estimates the noise robustly via the median absolute deviation (MAD) and
    flags pixels whose brightness exceeds
    ``median + hot_pixel_mad_sigma × robust_std``.  A minimum noise floor of
    1 ADU is applied so the threshold does not collapse to the median when the
    dark frame has no variation among its good pixels.

    Args:
        dark_array: Dark frame array (2D or 3D).

    Returns:
        Boolean 2D mask; True where a pixel is a hot-pixel candidate.
    """
    brightness = dark_array.max(axis=2) if dark_array.ndim == 3 else dark_array
    b = brightness.astype(np.float64)
    median = np.median(b)
    mad = np.median(np.abs(b - median))
    robust_std = max(1.4826 * mad, 1.0)
    return b > median + CFG.frame.hot_pixel_mad_sigma * robust_std


def subtract_master_dark(
    light_array: np.ndarray,
    dark_array: np.ndarray,
) -> np.ndarray:
    """Subtract a master dark from a light frame with saturating arithmetic.

    Unsigned integers clip at zero rather than wrapping.

    Args:
        light_array: Light frame array.
        dark_array: Master dark array; must match the dtype and shape of
            *light_array*.

    Returns:
        Result array with the same dtype as *light_array*.

    Raises:
        ValueError: If the dtype is unsupported.
    """
    if np.issubdtype(light_array.dtype, np.unsignedinteger):
        diff = light_array.astype(np.int32) - dark_array.astype(np.int32)
        limits = np.iinfo(light_array.dtype)
        return np.clip(diff, limits.min, limits.max).astype(light_array.dtype)

    msg = f"Unsupported dtype for subtraction: {light_array.dtype}"
    raise ValueError(msg)
