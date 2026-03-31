"""Public Frame implementation."""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import tifffile
import zarr

from celestack.exceptions import DownscaleError, TileReadError
from celestack.frame._backends import Backend, TiffBackend, backend_types
from celestack.frame._constants import CELESTACK_KEY, DEFAULT_TILE_SIZE
from celestack.frame._image_ops import (
    convert_bit_depth,
    downscale_by_block_average,
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
        self._backend = self._resolve_backend(self._path)
        info = self._backend.inspect(self._path)
        self._shape = info.shape
        self._bit_depth = info.bit_depth
        self._is_tiled = info.is_tiled

        celestack = info.celestack_metadata
        self._downscale_factor = celestack.downscale_factor
        self._timestamp_override = celestack.timestamp

        self._metadata = extract_exif_metadata(self._path)
        self._array_cache: np.ndarray | None = None

    @staticmethod
    def _resolve_backend(path: Path) -> Backend:
        """Select a parser backend based on the input file path."""
        for backend_type in backend_types():
            if backend_type.can_handle(path):
                return backend_type()
        msg = f"No backend found for: {path.suffix}"
        raise ValueError(msg)

    @property
    def path(self) -> Path:
        """Return the on-disk path represented by this frame."""
        return self._path

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
    def is_tiled(self) -> bool:
        """Return whether the underlying TIFF is stored in tiled layout."""
        return self._is_tiled

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
            self._array_cache = self._backend.load_array(self._path)
        return self._array_cache

    def unload(self) -> None:
        """Drop any cached pixel array to free memory."""
        self._array_cache = None

    def save(self, path: str | Path) -> Frame:
        """Save the frame as a tiled TIFF and return the saved Frame.

        Args:
            path: Destination file path.

        Returns:
            A new frame bound to the saved path.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if TiffBackend.can_handle(self.path) and self.is_tiled:
            shutil.copy2(self.path, target)
            return Frame(target)

        array = self.array
        photometric = "rgb" if array.ndim == 3 and array.shape[2] >= 3 else "minisblack"
        datetime_value = self._metadata.datetime
        tifffile.imwrite(
            target,
            data=array,
            tile=(DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE),
            compression="zlib",
            photometric=photometric,
            metadata=None,
            datetime=str(datetime_value) if datetime_value else None,
            extratags=build_tiff_extratags(self._metadata),
        )
        return Frame(target)

    def save_downscaled(
        self,
        path: str | Path,
        downscale_factor: int,
        bit_depth: int = 8,
        *,
        grayscale: bool = True,
    ) -> Frame:
        """Write a downscaled TIFF proxy and return it as a Frame.

        Args:
            path: Destination path for the proxy frame.
            downscale_factor: Integer linear downscale factor.
            bit_depth: Target output bit depth.
            grayscale: Whether to convert RGB input to grayscale.

        Returns:
            A new frame bound to the generated proxy path.

        Raises:
            DownscaleError: If called on a frame that is already downscaled.
            ValueError: If the downscale factor is invalid or timestamp is missing.
        """
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

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        working = self.array
        if grayscale:
            working = to_grayscale(working)

        downscaled = downscale_by_block_average(working, downscale_factor)
        converted = convert_bit_depth(downscaled, self.bit_depth, bit_depth)

        photometric = (
            "rgb" if converted.ndim == 3 and converted.shape[2] >= 3 else "minisblack"
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
            tile=(DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE),
            compression="zlib",
            photometric=photometric,
            metadata=metadata,
            datetime=str(self._metadata.datetime) if self._metadata.datetime else None,
            extratags=build_tiff_extratags(self._metadata),
        )
        return Frame(target)

    def read_tile(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Read a rectangular tile region from the image.

        Args:
            x: Left pixel coordinate.
            y: Top pixel coordinate.
            width: Tile width in pixels.
            height: Tile height in pixels.

        Returns:
            Pixel data for the requested region.

        Raises:
            ValueError: If bounds are invalid or outside image extents.
            TileReadError: If tile reading is requested for non-TIFF input.
        """
        if width <= 0 or height <= 0:
            msg = "width and height must be positive"
            raise ValueError(msg)
        if x < 0 or y < 0:
            msg = "x and y must be non-negative"
            raise ValueError(msg)

        h, w = self._shape[:2]
        if x + width > w or y + height > h:
            msg = "Requested tile is outside frame bounds"
            raise ValueError(msg)

        if not TiffBackend.can_handle(self.path):
            msg = "Tile reading is available only for TIFF files"
            raise TileReadError(msg)

        if self.is_tiled:
            with tifffile.TiffFile(self.path) as tif:
                store = tif.aszarr()
                z = zarr.open_array(store, mode="r")
                tile = np.asarray(z[y : y + height, x : x + width])
                store.close()
            return tile

        warnings.warn(
            f"Frame is not tiled: loading full image for tile read ({self.path.name})",
            RuntimeWarning,
            stacklevel=2,
        )
        return self.array[y : y + height, x : x + width]

    def plot(self, *, show_pixels: bool = False) -> go.Figure:
        """Create a Plotly figure with the frame as a background image.

        Args:
            show_pixels: Whether to plot with pixel hover data.

        Returns:
            A figure with axes in full-resolution pixel coordinates.
        """
        return plot_frame(
            self.array,
            title=self.path.name,
            bit_depth=self.bit_depth,
            downscale_factor=self.downscale_factor,
            show_pixels=show_pixels,
        )

    def __repr__(self) -> str:
        """Return a concise debug representation of the frame."""
        return f"Frame({str(self.path)!r}, downscale_factor={self.downscale_factor})"
