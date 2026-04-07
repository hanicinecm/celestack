"""Image backend implementations for Frame file handling."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import tifffile
from PIL import Image

from celestack.frame._metadata import (
    CelestackMetadata,
    FrameInfo,
    parse_celestack_metadata,
)


class Backend(Protocol):
    """Protocol for image format backends."""

    SUFFIXES: frozenset[str]

    def inspect(self, path: Path) -> FrameInfo:
        """Inspect basic frame characteristics without loading full pixels."""
        ...

    def load_array(self, path: Path) -> np.ndarray:
        """Load image pixels into a NumPy array."""
        ...


class EmptyBackend:
    """Fallback backend that raises errors for all operations."""

    SUFFIXES = frozenset()

    def inspect(self, path: Path) -> FrameInfo:
        msg = f"No backend available to inspect: {path}"
        raise ValueError(msg)

    def load_array(self, path: Path) -> np.ndarray:
        msg = f"No backend available to load: {path}"
        raise ValueError(msg)


class TiffBackend:
    """Backend for TIFF files."""

    SUFFIXES = frozenset({".tif", ".tiff"})

    def inspect(self, path: Path) -> FrameInfo:
        """Read TIFF page metadata used by Frame initialization."""
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            if not isinstance(page, tifffile.TiffPage):
                msg = f"Unexpected TIFF page type: {type(page)}"
                raise ValueError(msg)
            bit_depth = page.bitspersample
            if isinstance(bit_depth, tuple):
                bit_depth = int(bit_depth[0])
            if bit_depth is None:
                bit_depth = np.dtype(page.dtype).itemsize * 8
            info = FrameInfo(
                bit_depth=int(bit_depth),
                shape=tuple(int(v) for v in page.shape),
                celestack_metadata=parse_celestack_metadata(page),
            )
        return info

    def load_array(self, path: Path) -> np.ndarray:
        """Load TIFF pixels using tifffile."""
        return tifffile.imread(path)


class PillowBackend:
    """Backend for Pillow-supported non-TIFF formats."""

    SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})

    MODE_BIT_DEPTH = {
        "1": 1,
        "L": 8,
        "P": 8,
        "RGB": 8,
        "RGBA": 8,
        "CMYK": 8,
        "YCbCr": 8,
        "LAB": 8,
        "HSV": 8,
        "I;16": 16,
        "I;16L": 16,
        "I;16B": 16,
        "I;16N": 16,
    }

    def inspect(self, path: Path) -> FrameInfo:
        """Read image mode and shape metadata via Pillow."""
        with Image.open(path) as image:
            width, height = image.size
            bands = len(image.getbands())
            shape = (height, width) if bands == 1 else (height, width, bands)
            arr = np.asarray(image)
            bit_depth = self.MODE_BIT_DEPTH.get(image.mode)
            if bit_depth is None:
                bit_depth = arr.dtype.itemsize * 8
            return FrameInfo(
                bit_depth=bit_depth,
                shape=shape,
                celestack_metadata=CelestackMetadata(),
            )

    def load_array(self, path: Path) -> np.ndarray:
        """Load pixels via Pillow and convert to a NumPy array."""
        with Image.open(path) as image:
            return np.asarray(image)


def is_tiff_path(path: Path) -> bool:
    """Return True if the path has a TIFF file extension."""
    return path.suffix.lower() in TiffBackend.SUFFIXES


def get_backends() -> dict[str, Backend]:
    """Return a mapping from file suffix to backend instance."""
    backends: dict[str, Backend] = {}
    for cls in (TiffBackend, PillowBackend):
        instance = cls()
        for suffix in cls.SUFFIXES:
            if suffix in backends:
                msg = f"Duplicate backend for suffix: {suffix}"
                raise ValueError(msg)
            backends[suffix] = instance
    return backends
