"""Public Frame implementation."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

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
        self._dtype = info.dtype
        self._bit_depth = info.bit_depth

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
        frame._dtype = self._dtype
        frame._bit_depth = self._bit_depth
        frame._downscale_factor = self._downscale_factor
        frame._timestamp_override = self._timestamp_override
        frame._metadata = self._metadata
        frame._array_cache = (
            None if self._array_cache is None else self._array_cache.copy()
        )
        return frame

    @property
    def path(self) -> Path | None:
        """Return the on-disk path represented by this frame, if any."""
        return self._path

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the image dimensions discovered during inspection."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Return the pixel data type discovered during inspection."""
        return self._dtype

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
        """Return image pixels, loading them lazily on first access."""
        if self._array_cache is None:
            if self._path is None:
                msg = "In-memory frame has no backing path and no cached array"
                raise ValueError(msg)
            self._array_cache = self._backend.load_array(self._path)
        return self._array_cache

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

    def save(self, path: str | Path) -> Frame:
        """Save the frame as a TIFF and return the saved Frame.

        Existing TIFF sources are copied verbatim. Non-TIFF or detached sources are
        written as strip-layout TIFFs with EXIF metadata preserved.

        The array cache is conserved: if the pixel data was not loaded
        before the call, it is unloaded afterwards.

        Args:
            path: Destination file path (.tif or .tiff).

        Returns:
            A new frame bound to the saved path.

        Raises:
            ValueError: If *path* does not have a TIFF extension.
        """
        target = Path(path)
        if not is_tiff_path(target):
            msg = f"Output path must be a TIFF file, got: {target.suffix!r}"
            raise ValueError(msg)
        target.parent.mkdir(parents=True, exist_ok=True)

        if self.path is not None and is_tiff_path(self.path):
            shutil.copy2(self.path, target)
            return Frame(target)

        with self._conserve_cache():
            array = self.array
            photometric = (
                "rgb" if array.ndim == 3 and array.shape[2] >= 3 else "minisblack"
            )
            datetime_value = self._metadata.datetime
            tifffile.imwrite(
                target,
                data=array,
                compression="zlib",
                photometric=photometric,
                metadata=None,
                datetime=str(datetime_value) if datetime_value else None,
                extratags=build_tiff_extratags(self._metadata),
            )
        return Frame(target)

    def __sub__(self, other: object) -> Frame:
        """Return an in-memory frame produced by pixel-wise subtraction.

        The array cache is conserved on both operands: if the pixel data
        was not loaded before the call, it is unloaded afterwards.
        """
        if not isinstance(other, Frame):
            return NotImplemented

        self._validate_similarity(other)
        with self._conserve_cache(), other._conserve_cache():
            result = self._detached_copy()
            result._array_cache = subtract_arrays(self.array, other.array)
            result._shape = tuple(int(v) for v in result._array_cache.shape)
            result._dtype = np.dtype(result._array_cache.dtype)

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
        if self.dtype != other.dtype:
            msg = (
                f"Dtype mismatch: left frame has dtype {self.dtype}, "
                f"right frame has dtype {other.dtype}"
            )
            raise ValueError(msg)
        if self.bit_depth != other.bit_depth:
            msg = (
                f"Bit depth mismatch: left frame has bit depth {self.bit_depth}, "
                f"right frame has bit depth {other.bit_depth}"
            )
            raise ValueError(msg)
        if self.dtype == np.bool_:
            msg = "Boolean frames do not support subtraction"
            raise ValueError(msg)

    def save_downscaled(
        self,
        path: str | Path,
        downscale_factor: int,
        bit_depth: int = 8,
        *,
        grayscale: bool = True,
    ) -> Frame:
        """Write a downscaled TIFF proxy and return it as a Frame.

        The array cache is conserved: if the pixel data was not loaded
        before the call, it is unloaded afterwards.

        Args:
            path: Destination path for the proxy frame (.tif or .tiff).
            downscale_factor: Integer linear downscale factor.
            bit_depth: Target output bit depth.
            grayscale: Whether to convert RGB input to grayscale.

        Returns:
            A new frame bound to the generated proxy path.

        Raises:
            DownscaleError: If called on a frame that is already downscaled.
            ValueError: If the downscale factor is invalid, timestamp is
                missing, or *path* does not have a TIFF extension.
        """
        target = Path(path)
        if not is_tiff_path(target):
            msg = f"Output path must be a TIFF file, got: {target.suffix!r}"
            raise ValueError(msg)

        if self.downscale_factor != 1:
            msg = "Cannot downscale a proxy frame"
            raise DownscaleError(msg)
        if downscale_factor < 1:
            msg = "downscale_factor must be >= 1"
            raise ValueError(msg)

        timestamp = self.timestamp
        if timestamp is None:
            msg = "save_downscaled requires a non-None timestamp"
            raise ValueError(msg)

        target.parent.mkdir(parents=True, exist_ok=True)

        with self._conserve_cache():
            working = self.array
            if grayscale:
                working = to_grayscale(working)

            downscaled = downscale_by_block_average(working, downscale_factor)
            converted = convert_bit_depth(downscaled, self.bit_depth, bit_depth)

            photometric = (
                "rgb"
                if converted.ndim == 3 and converted.shape[2] >= 3
                else "minisblack"
            )
            metadata = {
                CELESTACK_KEY: {
                    "downscale_factor": int(downscale_factor),
                    "timestamp": float(timestamp),
                }
            }
            tifffile.imwrite(
                target,
                data=converted,
                compression="zlib",
                photometric=photometric,
                metadata=metadata,
                datetime=(
                    str(self._metadata.datetime) if self._metadata.datetime else None
                ),
                extratags=build_tiff_extratags(self._metadata),
            )

        return Frame(target)

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
                title=self.path.name if self.path is not None else "<memory>",
                bit_depth=self.bit_depth,
                downscale_factor=self.downscale_factor,
                show_pixels=show_pixels,
            )
        return figure

    def __repr__(self) -> str:
        """Return a concise debug representation of the frame."""
        return f"Frame({str(self.path)!r}, downscale_factor={self.downscale_factor})"
