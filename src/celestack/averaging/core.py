"""Public averaging API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from celestack.averaging._methods import get_methods
from celestack.frame._metadata import build_tiff_extratags
from celestack.frame.core import Frame
from celestack.progress import progress_factory

DEFAULT_BAND_HEIGHT = 256


def average_frames(
    frames: list[Frame],
    output_path: str | Path,
    method: str = "sigma_clip",
    *,
    reference_frame: Frame | None = None,
    band_height: int | None = DEFAULT_BAND_HEIGHT,
) -> Frame:
    """Average a list of frames into a single output frame.

    Processes the stack in horizontal row-bands so that only a small
    slice of each frame is held in memory at a time.

    Args:
        frames: Input frames to average (must all be TIFFs with the
            same shape and dtype).
        output_path: Destination path for the averaged TIFF.
        method: Averaging method — ``"median"``, ``"mean"``, or
            ``"sigma_clip"``.
        reference_frame: Optional frame whose EXIF metadata is
            inherited by the output file.
        band_height: Number of rows to process per band.  When
            ``None``, the entire array is averaged at
            once.

    Returns:
        A new frame bound to *output_path*.

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
    target = Path(output_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)

    if band_height is None:
        band_height = height

    n_bands = (height + band_height - 1) // band_height
    bar = progress_factory(n_bands, "Averaging")
    result_bands: list[np.ndarray] = []
    for y_start in range(0, height, band_height):
        y_end = min(y_start + band_height, height)
        band_slices = [
            f.read_tile(0, y_start, width, y_end, unload_array=True) for f in frames
        ]
        stack = np.stack(band_slices, axis=0)
        result_bands.append(combine(stack))
        bar.update()
    bar.close()

    result = np.concatenate(result_bands, axis=0)

    photometric = "rgb" if result.ndim == 3 and result.shape[2] >= 3 else "minisblack"
    write_kwargs: dict = {
        "compression": "zlib",
        "photometric": photometric,
    }
    if reference_frame is not None:
        meta = reference_frame.metadata
        if meta.datetime:
            write_kwargs["datetime"] = str(meta.datetime)
        extratags = build_tiff_extratags(meta)
        if extratags:
            write_kwargs["extratags"] = extratags

    tifffile.imwrite(target, data=result, **write_kwargs)
    return Frame(target)
