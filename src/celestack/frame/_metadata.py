"""Metadata models and parsers for frame files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import exifread
import tifffile

from celestack.frame._constants import CELESTACK_KEY


def as_float(value: Any) -> float | None:
    """Convert EXIF-like numeric values into a float when possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    num = getattr(value, "num", None)
    den = getattr(value, "den", None)
    if num is not None and den:
        return float(num) / float(den)

    text = str(value)
    if "/" in text:
        left, right = text.split("/", maxsplit=1)
        try:
            return float(left) / float(right)
        except ValueError:
            return None
    try:
        return float(text)
    except ValueError:
        return None


def parse_datetime_to_epoch(value: str | None) -> float | None:
    """Parse an EXIF datetime string into a UNIX timestamp."""
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S").timestamp()
    except ValueError:
        return None


def to_rational(value: float) -> tuple[int, int]:
    """Encode a float into a bounded rational tuple for TIFF tags."""
    fraction = Fraction(value).limit_denominator(1_000_000)
    return (fraction.numerator, fraction.denominator)


@dataclass(frozen=True)
class ExifMetadata:
    """Structured subset of EXIF metadata used by Frame."""

    datetime: str | None = None
    camera_model: str | None = None
    exposure: float | None = None
    iso: int | None = None
    focal_length: float | None = None


@dataclass(frozen=True)
class CelestackMetadata:
    """Structured Celestack metadata embedded into proxy TIFF files."""

    downscale_factor: int = 1
    timestamp: float | None = None


@dataclass(frozen=True)
class FrameInfo:
    """Metadata discovered during lightweight frame inspection."""

    bit_depth: int
    shape: tuple[int, ...]
    is_tiled: bool
    celestack_metadata: CelestackMetadata


def extract_exif_metadata(path: Path) -> ExifMetadata:
    """Read selected EXIF fields from an image file."""
    with path.open("rb") as f:
        tags = exifread.process_file(f, details=False)

    datetime_value: str | None = None
    camera_model_value: str | None = None
    exposure_value: float | None = None
    iso_value: int | None = None
    focal_length_value: float | None = None

    dt_tag = (
        tags.get("EXIF DateTimeOriginal")
        or tags.get("EXIF DateTimeDigitized")
        or tags.get("Image DateTime")
    )
    if dt_tag is not None:
        datetime_value = str(dt_tag)

    model_tag = tags.get("Image Model")
    if model_tag is not None:
        camera_model_value = str(model_tag)

    exposure_tag = tags.get("EXIF ExposureTime")
    exposure = as_float(exposure_tag)
    if exposure is not None:
        exposure_value = exposure

    iso_tag = tags.get("EXIF ISOSpeedRatings")
    if iso_tag is not None:
        try:
            iso_value = int(str(iso_tag))
        except ValueError:
            pass

    focal_tag = tags.get("EXIF FocalLength")
    focal = as_float(focal_tag)
    if focal is not None:
        focal_length_value = focal

    return ExifMetadata(
        datetime=datetime_value,
        camera_model=camera_model_value,
        exposure=exposure_value,
        iso=iso_value,
        focal_length=focal_length_value,
    )


def build_tiff_extratags(
    metadata: ExifMetadata,
) -> list[tuple[int, str, int, Any, bool]]:
    """Translate known metadata keys into TIFF extra tags."""
    extra: list[tuple[int, str, int, Any, bool]] = []
    model = metadata.camera_model
    if model:
        extra.append((272, "s", 0, str(model), True))

    exposure = metadata.exposure
    if exposure is not None:
        extra.append((33434, "2I", 1, to_rational(float(exposure)), True))

    iso = metadata.iso
    if iso is not None:
        extra.append((34855, "H", 1, int(iso), True))

    focal = metadata.focal_length
    if focal is not None:
        extra.append((37386, "2I", 1, to_rational(float(focal)), True))
    return extra


def parse_celestack_metadata(page: tifffile.TiffPage) -> CelestackMetadata:
    """Extract Celestack-specific metadata from TIFF ImageDescription."""
    tag = page.tags.get("ImageDescription")
    if tag is None:
        return CelestackMetadata()

    raw = tag.value
    if not isinstance(raw, str):
        return CelestackMetadata()

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return CelestackMetadata()

    if not isinstance(decoded, dict):
        return CelestackMetadata()

    block = decoded.get(CELESTACK_KEY)
    if isinstance(block, dict):
        downscale = block.get("downscale_factor")
        timestamp = block.get("timestamp")
        return CelestackMetadata(
            downscale_factor=int(downscale) if downscale is not None else 1,
            timestamp=float(timestamp) if timestamp is not None else None,
        )

    if "downscale_factor" in decoded or "timestamp" in decoded:
        downscale = decoded.get("downscale_factor")
        timestamp = decoded.get("timestamp")
        return CelestackMetadata(
            downscale_factor=int(downscale) if downscale is not None else 1,
            timestamp=float(timestamp) if timestamp is not None else None,
        )

    return CelestackMetadata()
