"""Shared test constants and helper functions.

This module provides reusable constants and utility functions for tests.
Conftest files should only contain fixtures; importable helpers live here.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile


EXIF_DATETIME = "2025:07:15 23:30:00"
EXIF_DATETIME_EPOCH = datetime.strptime(
    EXIF_DATETIME, "%Y:%m:%d %H:%M:%S"
).timestamp()
CAMERA_MAKE = "TestCorp"
CAMERA_MODEL = "TestCam X100"
LENS_MODEL = "TestLens 24mm f/1.4"
IMG_W, IMG_H = 64, 48


def write_tiff(
    path: Path,
    array: np.ndarray,
    *,
    photometric: str = "rgb",
    datetime_str: str | None = EXIF_DATETIME,
    extratags: list | None = None,
    metadata: dict | None = None,
) -> Path:
    """Write a TIFF with optional EXIF and Celestack metadata."""
    tags = extratags or []
    tifffile.imwrite(
        path,
        data=array,
        photometric=photometric,
        datetime=datetime_str,
        extratags=tags,
        metadata=metadata,
    )
    return path


def write_gray_tiff(path: Path, array: np.ndarray) -> Path:
    """Write a simple grayscale TIFF for testing."""
    tifffile.imwrite(path, data=array, photometric="minisblack")
    return path


def write_rgb_tiff(path: Path, array: np.ndarray) -> Path:
    """Write a simple RGB TIFF for testing."""
    tifffile.imwrite(path, data=array, photometric="rgb")
    return path
