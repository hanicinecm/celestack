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

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        """Return whether this backend can open the given path."""
        ...

    def inspect(self, path: Path) -> FrameInfo:
        """Inspect basic frame characteristics without loading full pixels."""
        ...

    def load_array(self, path: Path) -> np.ndarray:
        """Load image pixels into a NumPy array."""
        ...


class TiffBackend:
    """Backend for TIFF files."""

    SUFFIXES = frozenset({".tif", ".tiff"})

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        """Return whether the file extension is TIFF-like."""
        return path.suffix.lower() in cls.SUFFIXES

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
                is_tiled=bool(page.is_tiled),
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
        "I": 32,
        "F": 32,
    }

    @classmethod
    def can_handle(cls, path: Path) -> bool:
        """Return whether the file extension is handled by Pillow backend."""
        return path.suffix.lower() in cls.SUFFIXES

    def inspect(self, path: Path) -> FrameInfo:
        """Read image mode and shape metadata via Pillow."""
        with Image.open(path) as image:
            width, height = image.size
            bands = len(image.getbands())
            shape = (height, width) if bands == 1 else (height, width, bands)
            bit_depth = self.MODE_BIT_DEPTH.get(image.mode)
            if bit_depth is None:
                bit_depth = np.asarray(image).dtype.itemsize * 8
            return FrameInfo(
                bit_depth=bit_depth,
                shape=shape,
                is_tiled=False,
                celestack_metadata=CelestackMetadata(),
            )

    def load_array(self, path: Path) -> np.ndarray:
        """Load pixels via Pillow and convert to a NumPy array."""
        with Image.open(path) as image:
            return np.asarray(image)


def backend_types() -> tuple[type[Backend], ...]:
    """Return available backend classes in resolution order."""
    return (TiffBackend, PillowBackend)
