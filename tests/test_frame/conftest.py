"""Fixtures for the Frame test sub-package."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

from celestack.frame._metadata import CELESTACK_KEY
from tests.utils import (
    CAMERA_MAKE,
    CAMERA_MODEL,
    EXIF_DATETIME,
    IMG_H,
    IMG_W,
    LENS_MODEL,
    write_tiff,
)


@pytest.fixture()
def tmp_rgb_tiff(tmp_path: Path) -> Path:
    """64x48 RGB 16-bit TIFF with EXIF datetime and camera model."""
    array = np.random.default_rng(42).integers(
        0, 65535, (IMG_H, IMG_W, 3), dtype=np.uint16
    )
    path = tmp_path / "rgb.tif"
    extratags = [
        (271, "s", 0, CAMERA_MAKE, True),
        (272, "s", 0, CAMERA_MODEL, True),
        (33434, "2I", 1, (30, 1), True),
        (33437, "2I", 1, (14, 10), True),
        (34855, "H", 1, 1600, True),
        (37386, "2I", 1, (24, 1), True),
        (42036, "s", 0, LENS_MODEL, True),
    ]
    return write_tiff(path, array, extratags=extratags)


@pytest.fixture()
def tmp_gray_tiff(tmp_path: Path) -> Path:
    """64x48 grayscale 16-bit TIFF."""
    array = np.random.default_rng(43).integers(
        0, 65535, (IMG_H, IMG_W), dtype=np.uint16
    )
    path = tmp_path / "gray.tif"
    return write_tiff(path, array, photometric="minisblack")


@pytest.fixture()
def tmp_rgb_jpeg(tmp_path: Path) -> Path:
    """64x48 RGB 8-bit JPEG with EXIF datetime and camera model."""
    import piexif

    array = np.random.default_rng(44).integers(
        0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8
    )
    path = tmp_path / "image.jpg"
    img = Image.fromarray(array, mode="RGB")

    exif_dict = {
        "0th": {
            piexif.ImageIFD.Make: CAMERA_MAKE.encode(),
            piexif.ImageIFD.Model: CAMERA_MODEL.encode(),
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: EXIF_DATETIME.encode(),
            piexif.ExifIFD.ExposureTime: (30, 1),
            piexif.ExifIFD.FNumber: (14, 10),
            piexif.ExifIFD.ISOSpeedRatings: 1600,
            piexif.ExifIFD.FocalLength: (24, 1),
            piexif.ExifIFD.LensModel: LENS_MODEL.encode(),
        },
    }
    exif_bytes = piexif.dump(exif_dict)
    img.save(path, exif=exif_bytes)
    return path


@pytest.fixture()
def tmp_no_metadata_tiff(tmp_path: Path) -> Path:
    """TIFF with no EXIF data at all."""
    array = np.random.default_rng(45).integers(
        0, 65535, (IMG_H, IMG_W), dtype=np.uint16
    )
    path = tmp_path / "no_meta.tif"
    return write_tiff(path, array, photometric="minisblack", datetime_str=None)


@pytest.fixture()
def tmp_tiled_tiff(tmp_path: Path) -> Path:
    """Tiled TIFF with tile-based layout."""
    array = np.random.default_rng(46).integers(
        0, 65535, (IMG_H, IMG_W), dtype=np.uint16
    )
    path = tmp_path / "tiled.tif"
    tile_size = 64
    tifffile.imwrite(
        path,
        data=array,
        tile=(tile_size, tile_size),
        compression="zlib",
        photometric="minisblack",
    )
    return path


@pytest.fixture()
def tmp_bool_mask_tiff(tmp_path: Path) -> Path:
    """Boolean mask saved as a TIFF."""
    rng = np.random.default_rng(47)
    array = rng.choice([True, False], size=(IMG_H, IMG_W))
    path = tmp_path / "mask.tif"
    return write_tiff(path, array.astype(np.uint8), photometric="minisblack")


@pytest.fixture()
def tmp_celestack_tiff(tmp_path: Path) -> Path:
    """TIFF with embedded Celestack metadata (downscale_factor, timestamp)."""
    array = np.random.default_rng(48).integers(
        0, 255, (IMG_H // 4, IMG_W // 4), dtype=np.uint8
    )
    path = tmp_path / "proxy.tif"
    celestack_meta = {
        CELESTACK_KEY: {
            "downscale_factor": 4,
            "timestamp": 1752620400.0,
        }
    }
    return write_tiff(
        path,
        array,
        photometric="minisblack",
        metadata=celestack_meta,
    )
