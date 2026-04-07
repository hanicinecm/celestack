"""Public Frame implementation."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, overload

import numpy as np
import plotly.graph_objects as go
import tifffile

from celestack.exceptions import DownscaleError
from celestack.frame._backends import get_backends, is_tiff_path
from celestack.frame._constants import CELESTACK_KEY
from celestack.frame._image_ops import (
    convert_bit_depth,
    downscale_by_block_average,
    subtract_arrays,
    to_grayscale,
)
from celestack.frame._metadata import (
    ExifMetadata,
    build_tiff_extratags,
    extract_exif_metadata,
    parse_datetime_to_epoch,
)
from celestack.frame._plotting import plot_frame


class Frame:
    """Universal image container for Celestack workflow artifacts."""

    def __init__(self, path: str | Path) -> None:
        """Create a frame from an image path.

        Args:
            path: Source image path.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the image format is unsupported.
        """
        resolved = Path(path)
        if not resolved.exists():
            msg = f"Frame not found: {resolved}"
            raise FileNotFoundError(msg)

        self._path = resolved
        backends = get_backends()
        suffix = self._path.suffix.lower()
        if suffix not in backends:
            msg = f"No backend found for: {self._path.suffix}"
            raise ValueError(msg)
        self._backend = backends[suffix]
        info = self._backend.inspect(self._path)
        self._shape = info.shape
        self._bit_depth = info.bit_depth
        if self._bit_depth not in {8, 16}:
            msg = f"Unsupported bit depth: {self._bit_depth} (supported: 8, 16)"
            raise ValueError(msg)

        celestack = info.celestack_metadata
        self._downscale_factor = celestack.downscale_factor
        self._timestamp_override = celestack.timestamp

        self._metadata = extract_exif_metadata(self._path)
        self._array_cache: np.ndarray | None = None

    def _detached_copy(self) -> Frame:
        """Return a pathless in-memory copy of the frame state."""
        frame = type(self).__new__(type(self))
        frame._path = None
        frame._backend = self._backend
        frame._shape = self._shape
        frame._bit_depth = self._bit_depth
        frame._downscale_factor = self._downscale_factor
        frame._timestamp_override = self._timestamp_override
        frame._metadata = self._metadata
        frame._array_cache = (
            None if self._array_cache is None else self._array_cache.copy()
        )
        return frame

    @property
    def path(self) -> Path:
        """Return the on-disk path represented by this frame.

        Raises:
            AttributeError: If the frame is not backed by a file.
        """
        if self._path is None:
            msg = "Frame is not backed by a file"
            raise AttributeError(msg)
        return self._path

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the image dimensions discovered during inspection."""
        return self._shape

    @property
    def bit_depth(self) -> int:
        """Return detected source bit depth."""
        return self._bit_depth

    @property
    def downscale_factor(self) -> int:
        """Return the proxy downscale factor stored in frame metadata."""
        return self._downscale_factor

    @property
    def metadata(self) -> ExifMetadata:
        """Return the extracted EXIF metadata."""
        return self._metadata

    @property
    def timestamp(self) -> float | None:
        """Return frame timestamp from override or EXIF metadata."""
        if self._timestamp_override is not None:
            return self._timestamp_override
        return parse_datetime_to_epoch(self._metadata.datetime)

    @timestamp.setter
    def timestamp(self, value: float | None) -> None:
        """Override frame timestamp for downstream processing."""
        self._timestamp_override = None if value is None else float(value)

    @property
    def array(self) -> np.ndarray:
        """Return image pixels, loading them lazily on first access.

        Raises:
            ValueError: If the frame has no backing path and no cached array.
            TypeError: If the loaded array dtype does not match the expected
                dtype for the frame's bit depth.
        """
        if self._array_cache is None:
            if self._path is None:
                msg = "In-memory frame has no backing path and no cached array"
                raise ValueError(msg)
            loaded = self._backend.load_array(self._path)
            expected = np.dtype(f"uint{self._bit_depth}")
            if loaded.dtype != expected:
                msg = (
                    f"Loaded array dtype {loaded.dtype!r} does not match "
                    f"expected {expected!r} for bit depth {self._bit_depth}"
                )
                raise TypeError(msg)
            self._array_cache = loaded
        return self._array_cache

    def load(self) -> None:
        """Load the pixel array into memory explicitly."""
        _ = self.array

    def unload(self) -> None:
        """Drop any cached pixel array to free memory."""
        self._array_cache = None

    @contextmanager
    def _conserve_cache(self) -> Iterator[None]:
        """Restore the array cache to its prior state on exit.

        If the pixel data was not loaded before entering the block,
        it is unloaded when the block exits.
        """
        was_loaded = self._array_cache is not None
        yield
        if not was_loaded:
            self.unload()

    def save_as(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Write the frame to a TIFF file and bind this instance to that path.

        Existing TIFF sources are copied verbatim. Non-TIFF or detached sources
        are written as strip-layout TIFFs with EXIF metadata preserved. Proxy
        frames (downscale_factor > 1) have Celestack metadata embedded.

        The array cache is conserved: if the pixel data was not loaded before
        the call, it is unloaded afterwards.

        Args:
            path: Destination file path (.tif or .tiff).
            overwrite: If False (default), raise if the destination exists.

        Raises:
            ValueError: If *path* does not have a TIFF extension.
            FileExistsError: If the destination exists and *overwrite* is False.
        """
        target = Path(path)
        if not is_tiff_path(target):
            msg = f"Output path must be a TIFF file, got: {target.suffix!r}"
            raise ValueError(msg)
        if target.exists() and not overwrite:
            msg = f"File already exists: {target}"
            raise FileExistsError(msg)
        target.parent.mkdir(parents=True, exist_ok=True)

        if self._path is not None and is_tiff_path(self._path):
            shutil.copy2(self._path, target)
        else:
            with self._conserve_cache():
                array = self.array
                photometric = (
                    "rgb" if array.ndim == 3 and array.shape[2] >= 3 else "minisblack"
                )
                datetime_value = self._metadata.datetime

                if self._downscale_factor > 1:
                    celestack_block: dict = {
                        "downscale_factor": int(self._downscale_factor)
                    }
                    if self.timestamp is not None:
                        celestack_block["timestamp"] = float(self.timestamp)
                    metadata = {CELESTACK_KEY: celestack_block}
                else:
                    metadata = None

                tifffile.imwrite(
                    target,
                    data=array,
                    compression="zlib",
                    photometric=photometric,
                    metadata=metadata,
                    datetime=str(datetime_value) if datetime_value else None,
                    extratags=build_tiff_extratags(self._metadata),
                )

        self._path = target
        backends = get_backends()
        self._backend = backends[target.suffix.lower()]

    @overload
    def subtract(self, other: Frame, *, inplace: Literal[False] = ...) -> Frame: ...

    @overload
    def subtract(self, other: Frame, *, inplace: Literal[True]) -> None: ...

    def subtract(self, other: Frame, *, inplace: bool = False) -> Frame | None:
        """Subtract *other* from this frame pixel-wise, saturating at zero.

        The array cache is conserved on both operands when not in-place: if
        pixel data was not loaded before the call, it is unloaded afterwards.
        In-place mode conserves the cache on *other* only; self's cache is
        replaced with the result and self is detached from its backing path.

        Args:
            other: Frame to subtract. Must match shape, dtype, bit depth, and
                downscale factor.
            inplace: If True, update self in place and return None. If False
                (default), return a new detached Frame.

        Returns:
            A new detached Frame when *inplace* is False, otherwise None.

        Raises:
            ValueError: If the frames are incompatible.
        """
        self._validate_similarity(other)

        if inplace:
            with other._conserve_cache():
                result_array = subtract_arrays(self.array, other.array)
            self._array_cache = result_array
            self._path = None
            return None

        with self._conserve_cache(), other._conserve_cache():
            result = self._detached_copy()
            result._array_cache = subtract_arrays(self.array, other.array)
            result._shape = tuple(int(v) for v in result._array_cache.shape)
        return result

    def _validate_similarity(self, other: Frame) -> None:
        """Validate that two frames can be subtracted safely."""
        if self.downscale_factor != other.downscale_factor:
            msg = (
                "Downscale factor mismatch: left frame has downscale factor "
                f"{self.downscale_factor}, right frame has downscale factor "
                f"{other.downscale_factor}"
            )
            raise ValueError(msg)
        if self.shape != other.shape:
            msg = (
                f"Shape mismatch: left frame has shape {self.shape}, "
                f"right frame has shape {other.shape}"
            )
            raise ValueError(msg)
        if self.bit_depth != other.bit_depth:
            msg = (
                f"Bit depth mismatch: left frame has bit depth {self.bit_depth}, "
                f"right frame has bit depth {other.bit_depth}"
            )
            raise ValueError(msg)

    def downscaled_copy(
        self,
        downscale_factor: int,
        bit_depth: int = 8,
        *,
        grayscale: bool = True,
    ) -> Frame:
        """Return a detached downscaled copy of this frame.

        The result is held in memory. Call :meth:`save_as` on the returned
        frame to persist it to disk.

        The array cache is conserved: if the pixel data was not loaded before
        the call, it is unloaded afterwards.

        Args:
            downscale_factor: Integer linear downscale factor (>= 1).
            bit_depth: Target output bit depth.
            grayscale: Whether to convert RGB input to grayscale.

        Returns:
            A new detached Frame with the downscaled pixel data.

        Raises:
            DownscaleError: If called on a frame that is already downscaled.
            ValueError: If *downscale_factor* is less than 1.
        """
        if self.downscale_factor != 1:
            msg = "Cannot downscale a proxy frame"
            raise DownscaleError(msg)
        if downscale_factor < 1:
            msg = "downscale_factor must be >= 1"
            raise ValueError(msg)

        with self._conserve_cache():
            working = self.array
            if grayscale:
                working = to_grayscale(working)
            downscaled = downscale_by_block_average(working, downscale_factor)
            converted = convert_bit_depth(downscaled, self.bit_depth, bit_depth)

        result = self._detached_copy()
        result._array_cache = converted
        result._shape = tuple(int(v) for v in converted.shape)
        result._bit_depth = bit_depth
        result._downscale_factor = downscale_factor
        return result

    def read_tile(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        *,
        unload_array: bool = False,
    ) -> np.ndarray:
        """Read a rectangular region from the image.

        Args:
            x0: Left pixel coordinate.
            y0: Top pixel coordinate.
            x1: Right pixel coordinate (exclusive).
            y1: Bottom pixel coordinate (exclusive).
            unload_array: Drop cached array after slicing.

        Returns:
            Pixel data for the requested region.

        Raises:
            ValueError: If bounds are invalid or outside image extents.
        """
        if x1 <= x0 or y1 <= y0:
            msg = "x1 must be greater than x0 and y1 must be greater than y0"
            raise ValueError(msg)
        if x0 < 0 or y0 < 0:
            msg = "x0 and y0 must be non-negative"
            raise ValueError(msg)

        h, w = self._shape[:2]
        if x1 > w or y1 > h:
            msg = "Requested tile is outside frame bounds"
            raise ValueError(msg)

        tile = self.array[y0:y1, x0:x1].copy()
        if unload_array:
            self.unload()

        return tile

    def plot(self, *, show_pixels: bool = False) -> go.Figure:
        """Create a Plotly figure with the frame as a background image.

        The array cache is conserved: if the pixel data was not loaded
        before the call, it is unloaded afterwards.

        Args:
            show_pixels: Whether to plot with pixel hover data.

        Returns:
            A figure with axes in full-resolution pixel coordinates.
        """
        with self._conserve_cache():
            figure = plot_frame(
                self.array,
                title=self._path.name if self._path is not None else "<memory>",
                bit_depth=self.bit_depth,
                downscale_factor=self.downscale_factor,
                show_pixels=show_pixels,
            )
        return figure

    def __repr__(self) -> str:
        """Return a concise debug representation of the frame."""
        path_str = str(self._path) if self._path is not None else "<detached>"
        return f"Frame({path_str!r}, downscale_factor={self.downscale_factor})"
