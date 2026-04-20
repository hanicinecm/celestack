"""Image backend implementations for Frame file handling."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import tifffile
from PIL import Image

from celestack.frame._metadata import FrameInfo


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
            dtype = str(np.dtype(page.dtype))
            return FrameInfo(
                dtype=dtype,
                shape=tuple(int(v) for v in page.shape),
            )

    def load_array(self, path: Path) -> np.ndarray:
        """Load TIFF pixels using tifffile."""
        return tifffile.imread(path)


class PillowBackend:
    """Backend for Pillow-supported non-TIFF formats."""

    SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})

    MODE_DTYPE = {
        "1": "bool",
        "L": "uint8",
        "P": "uint8",
        "RGB": "uint8",
        "RGBA": "uint8",
        "CMYK": "uint8",
        "YCbCr": "uint8",
        "LAB": "uint8",
        "HSV": "uint8",
        "I;16": "uint16",
        "I;16L": "uint16",
        "I;16B": "uint16",
        "I;16N": "uint16",
    }

    def inspect(self, path: Path) -> FrameInfo:
        """Read image mode and shape metadata via Pillow."""
        with Image.open(path) as image:
            width, height = image.size
            bands = len(image.getbands())
            shape = (height, width) if bands == 1 else (height, width, bands)
            dtype = self.MODE_DTYPE.get(image.mode)
            if dtype is None:
                arr = np.asarray(image)
                dtype = str(arr.dtype)
            return FrameInfo(dtype=dtype, shape=shape)

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
