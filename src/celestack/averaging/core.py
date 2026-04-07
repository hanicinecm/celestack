"""Public averaging API."""

from __future__ import annotations

import numpy as np

from celestack.averaging._methods import get_methods
from celestack.frame.core import Frame
from celestack.progress import progress_factory

DEFAULT_BAND_HEIGHT = 256


def average_frames(
    frames: list[Frame],
    method: str = "sigma_clip",
    *,
    reference_frame: Frame | None = None,
    band_height: int | None = DEFAULT_BAND_HEIGHT,
) -> Frame:
    """Average a list of frames into a single detached in-memory frame.

    Processes the stack in horizontal row-bands so that only a small
    slice of each frame is held in memory at a time.

    Args:
        frames: Input frames to average (must all be TIFFs with the
            same shape and dtype).
        method: Averaging method — ``"median"``, ``"mean"``, or
            ``"sigma_clip"``.
        reference_frame: Optional frame whose metadata is inherited by
            the output frame. Defaults to the first frame in *frames*.
        band_height: Number of rows to process per band. When
            ``None``, the entire array is averaged at once.

    Returns:
        A new detached Frame with the averaged pixel data and metadata
        inherited from *reference_frame* (or the first input frame).
        Call :meth:`~celestack.frame.Frame.save_as` on the result to
        persist it to disk.

    Raises:
        ValueError: If *frames* is empty, shapes/dtypes are inconsistent,
            or *method* is unknown.
    """
    if not frames:
        msg = "frames must be non-empty"
        raise ValueError(msg)

    methods = get_methods()
    if method not in methods:
        msg = f"Unknown averaging method: {method!r}"
        raise ValueError(msg)
    combine = methods[method]

    ref = frames[0]
    for f in frames[1:]:
        if f.shape != ref.shape:
            msg = f"Shape mismatch: {f!r} has shape {f.shape}, expected {ref.shape}"
            raise ValueError(msg)
        if f.bit_depth != ref.bit_depth:
            msg = (
                f"Bit depth mismatch: {f!r} has {f.bit_depth}, expected {ref.bit_depth}"
            )
            raise ValueError(msg)

    height, width = ref.shape[0], ref.shape[1]

    if band_height is None:
        band_height = height

    # float32 is sufficient for uint8 and uint16: both fit exactly within
    # float32's 24-bit mantissa (max uint16 value 65535 << 2^24).
    input_dtype = np.dtype(f"uint{ref.bit_depth}")
    float_dtype = np.float32

    n_bands = (height + band_height - 1) // band_height
    bar = progress_factory(n_bands, "Averaging")
    result_bands: list[np.ndarray] = []
    for y_start in range(0, height, band_height):
        y_end = min(y_start + band_height, height)
        band_slices = [
            f.read_tile(0, y_start, width, y_end, unload_array=True) for f in frames
        ]
        stack = np.stack(band_slices, axis=0).astype(float_dtype)
        result_bands.append(combine(stack).astype(input_dtype))
        bar.update()
    bar.close()

    result_array = np.concatenate(result_bands, axis=0)

    meta_source = reference_frame if reference_frame is not None else frames[0]
    out_frame = meta_source._detached_copy()
    out_frame._array_cache = result_array
    out_frame._shape = tuple(int(v) for v in result_array.shape)
    return out_frame
