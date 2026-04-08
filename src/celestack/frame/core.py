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
from celestack.frame._backends import Backend, EmptyBackend, get_backends, is_tiff_path
from celestack.frame._constants import CELESTACK_KEY
from celestack.frame._image_ops import (
    build_dark_bad_pixel_mask,
    convert_bit_depth,
    downscale_by_block_average,
    interpolate_bad_pixels,
    subtract_master_dark,
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
        self._path: Path | None = Path(path)
        self._backend: Backend = EmptyBackend()
        self._attach(path)

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

    def _attach(self, path: str | Path) -> None:
        """Attach the frame to a backing path, making it a disk-backed frame.

        Args:
            path: The path to the image file to back this frame with.

        Raises:
            FileNotFoundError: If the path does not exist.
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

    def _detach(self) -> None:
        """Detach the frame from its backing path, making it an in-memory frame."""
        self._path = None
        self._backend = EmptyBackend()

    def _detached_copy(self) -> Frame:
        """Return a pathless in-memory copy of the frame state."""
        # Make one-to-one copy:
        frame = type(self).__new__(type(self))
        frame._path = self._path
        frame._backend = self._backend
        frame._shape = self._shape
        frame._bit_depth = self._bit_depth
        frame._downscale_factor = self._downscale_factor
        frame._timestamp_override = self._timestamp_override
        frame._metadata = self._metadata
        frame._array_cache = (
            None if self._array_cache is None else self._array_cache.copy()
        )

        # Detach the copy from any backing path:
        frame._detach()

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
        if self._path is None:
            msg = "Cannot unload array for an in-memory frame with no backing path"
            raise ValueError(msg)
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

        self._attach(target)

    def subtract_dark(self, master_dark: Frame) -> None:
        """Subtract a master dark from this frame in place.

        Performs saturating subtraction (unsigned integers clip at zero).
        Self is detached from its backing path after the operation.

        Args:
            master_dark: Master dark frame. Must match shape, dtype, bit
                depth, and downscale factor.

        Raises:
            ValueError: If the frames are incompatible.
        """
        self._validate_similarity(master_dark)
        with master_dark._conserve_cache():
            result_array = subtract_master_dark(self.array, master_dark.array)
        self._array_cache = result_array
        self._detach()

    def interpolate_bad_pixels(self, master_dark: Frame) -> None:
        """Replace hot pixels in this frame using the master dark as a bad pixel map.

        Identifies hot pixels in *master_dark* via a MAD-based threshold and
        replaces the corresponding positions in this frame with the median of
        their valid 8-connected neighbors. Self is detached from its backing
        path after the operation.

        Args:
            master_dark: Master dark frame used solely for hot-pixel
                detection. Must match shape, dtype, bit depth, and downscale
                factor.

        Raises:
            ValueError: If the frames are incompatible.
        """
        self._validate_similarity(master_dark)
        with master_dark._conserve_cache():
            mask = build_dark_bad_pixel_mask(master_dark.array)
            result_array = interpolate_bad_pixels(self.array, mask)
        self._array_cache = result_array
        self._detach()

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
