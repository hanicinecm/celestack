"""Public Frame implementation."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import tifffile

from celestack.frame._backends import Backend, EmptyBackend, get_backends, is_tiff_path
from celestack.frame._image_ops import (
    build_dark_bad_pixel_mask,
    interpolate_bad_pixels,
    subtract_master_dark,
)
from celestack.frame._metadata import (
    ExifMetadata,
    build_tiff_extratags,
    extract_exif_metadata,
    parse_datetime_to_epoch,
)
from celestack.frame._plotting import plot_frame

_SUPPORTED_DTYPES = frozenset({"uint8", "uint16"})
_DTYPE_TO_BIT_DEPTH = {"uint8": 8, "uint16": 16}


class Frame:
    """Universal full-resolution image container."""

    def __init__(self, path: str | Path) -> None:
        """Create a frame from an image path.

        Args:
            path: Source image path.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the image format or dtype is unsupported.
        """
        self._path: Path | None = Path(path)
        self._backend: Backend = EmptyBackend()
        self._attach(path)

        info = self._backend.inspect(self._path)
        self._shape = info.shape
        self._dtype = info.dtype
        if self._dtype not in _SUPPORTED_DTYPES:
            msg = (
                f"Unsupported dtype: {self._dtype!r} "
                f"(supported: {sorted(_SUPPORTED_DTYPES)})"
            )
            raise ValueError(msg)

        self._timestamp_override: float | None = None
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
        frame = type(self).__new__(type(self))
        frame._path = self._path
        frame._backend = self._backend
        frame._shape = self._shape
        frame._dtype = self._dtype
        frame._timestamp_override = self._timestamp_override
        frame._metadata = self._metadata
        frame._array_cache = (
            None if self._array_cache is None else self._array_cache.copy()
        )

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
    def dtype(self) -> str:
        """Return the pixel dtype as a string (``"uint8"`` or ``"uint16"``)."""
        return self._dtype

    @property
    def bit_depth(self) -> int:
        """Return the bit depth derived from :attr:`dtype`."""
        return _DTYPE_TO_BIT_DEPTH[self._dtype]

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
                dtype inspected from the backing file.
        """
        if self._array_cache is None:
            if self._path is None:
                msg = "In-memory frame has no backing path and no cached array"
                raise ValueError(msg)
            loaded = self._backend.load_array(self._path)
            expected = np.dtype(self._dtype)
            if loaded.dtype != expected:
                msg = (
                    f"Loaded array dtype {loaded.dtype!r} does not match "
                    f"expected {expected!r}"
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
        are written as strip-layout TIFFs with EXIF metadata preserved.

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

                tifffile.imwrite(
                    target,
                    data=array,
                    compression="zlib",
                    photometric=photometric,
                    datetime=str(datetime_value) if datetime_value else None,
                    extratags=build_tiff_extratags(self._metadata),
                )

        self._attach(target)

    def subtract_dark(self, master_dark: Frame) -> None:
        """Subtract a master dark from this frame in place.

        Performs saturating subtraction (unsigned integers clip at zero).
        Self is detached from its backing path after the operation.

        Args:
            master_dark: Master dark frame. Must match shape and dtype.

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
                detection. Must match shape and dtype.

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
        """Validate that two frames can be combined safely."""
        if self.shape != other.shape:
            msg = (
                f"Shape mismatch: left frame has shape {self.shape}, "
                f"right frame has shape {other.shape}"
            )
            raise ValueError(msg)
        if self.dtype != other.dtype:
            msg = (
                f"Dtype mismatch: left frame has dtype {self.dtype!r}, "
                f"right frame has dtype {other.dtype!r}"
            )
            raise ValueError(msg)

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
            A figure with axes in the frame's pixel coordinates.
        """
        with self._conserve_cache():
            figure = plot_frame(
                self.array,
                title=self._path.name if self._path is not None else "<detached>",
                bit_depth=self.bit_depth,
                show_pixels=show_pixels,
            )
        return figure

    def __repr__(self) -> str:
        """Return a concise debug representation of the frame."""
        path_str = str(self._path) if self._path is not None else "<detached>"
        return f"Frame({path_str!r}, dtype={self._dtype!r})"
