"""Memory-efficient frame averaging via row-band processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile
import zarr

from celestack.frame.core import Frame

DEFAULT_BAND_HEIGHT = 256


def average_frames(
    frames: list[Frame],
    output_path: str | Path,
    method: str = "median",
    *,
    band_height: int = DEFAULT_BAND_HEIGHT,
) -> Frame:
    """Average a list of frames into a single output frame.

    Processes the stack in horizontal row-bands so that only a small
    slice of each frame is held in memory at a time.  Works with any
    TIFF layout (tiled or strip).

    Args:
        frames: Input frames to average (must all be TIFFs with the
            same shape and dtype).
        output_path: Destination path for the averaged TIFF.
        method: Averaging method — ``"median"``, ``"mean"``, or
            ``"sigma_clip"``.
        band_height: Number of rows to process per band.

    Returns:
        A new frame bound to *output_path*.

    Raises:
        ValueError: If *frames* is empty, shapes/dtypes are inconsistent,
            or *method* is unknown.
    """
    if not frames:
        msg = "frames must be non-empty"
        raise ValueError(msg)

    methods = {"mean": _mean, "median": _median, "sigma_clip": _sigma_clip}
    if method not in methods:
        msg = f"Unknown averaging method: {method!r}"
        raise ValueError(msg)
    combine = methods[method]

    ref_shape, ref_dtype = _inspect_tiff(frames[0])
    for f in frames[1:]:
        shape, dtype = _inspect_tiff(f)
        if shape != ref_shape:
            msg = (
                f"Shape mismatch: {f.path.name} has shape {shape}, expected {ref_shape}"
            )
            raise ValueError(msg)
        if dtype != ref_dtype:
            msg = (
                f"Dtype mismatch: {f.path.name} has dtype {dtype}, expected {ref_dtype}"
            )
            raise ValueError(msg)

    height = ref_shape[0]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    result_bands: list[np.ndarray] = []
    for y_start in range(0, height, band_height):
        y_end = min(y_start + band_height, height)
        band_slices = _read_band(frames, y_start, y_end)
        stack = np.stack(band_slices, axis=0)
        result_bands.append(combine(stack))

    result = np.concatenate(result_bands, axis=0)

    photometric = "rgb" if result.ndim == 3 and result.shape[2] >= 3 else "minisblack"
    tifffile.imwrite(
        target,
        data=result,
        compression="zlib",
        photometric=photometric,
    )
    return Frame(target)


def _inspect_tiff(frame: Frame) -> tuple[tuple[int, ...], np.dtype]:
    """Return the array shape and dtype of a TIFF without loading it."""
    with tifffile.TiffFile(frame.path) as tif:
        store = tif.aszarr()
        z = zarr.open_array(store, mode="r")
        shape = z.shape
        dtype = z.dtype
        store.close()
    return tuple(int(d) for d in shape), dtype


def _read_band(frames: list[Frame], y_start: int, y_end: int) -> list[np.ndarray]:
    """Read a horizontal band from each frame via zarr slicing."""
    bands: list[np.ndarray] = []
    for frame in frames:
        with tifffile.TiffFile(frame.path) as tif:
            store = tif.aszarr()
            z = zarr.open_array(store, mode="r")
            band = np.asarray(z[y_start:y_end])
            store.close()
        bands.append(band)
    return bands


def _mean(stack: np.ndarray) -> np.ndarray:
    """Combine along axis 0 using the arithmetic mean."""
    return np.mean(stack, axis=0).astype(stack.dtype)


def _median(stack: np.ndarray) -> np.ndarray:
    """Combine along axis 0 using the median."""
    return np.median(stack, axis=0).astype(stack.dtype)


def _sigma_clip(stack: np.ndarray, kappa: float = 3.0) -> np.ndarray:
    """Combine along axis 0 using sigma-clipped mean.

    Args:
        stack: Array with shape ``(N, H, W, ...)`` where *N* is the
            number of frames.
        kappa: Number of standard deviations for clipping.

    Returns:
        Averaged array with shape ``(H, W, ...)``.
    """
    mean = np.mean(stack, axis=0, keepdims=True)
    std = np.std(stack, axis=0, keepdims=True)
    mask = np.abs(stack - mean) <= kappa * std
    clipped_sum = np.where(mask, stack, 0).sum(axis=0)
    clipped_count = mask.sum(axis=0)
    safe_count = np.where(clipped_count > 0, clipped_count, 1)
    result = np.where(
        clipped_count > 0,
        clipped_sum / safe_count,
        mean.squeeze(axis=0),
    )
    return result.astype(stack.dtype)
