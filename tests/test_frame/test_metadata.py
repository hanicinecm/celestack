"""Tests for frame metadata parsing and serialization."""

from typing import cast

import numpy as np
import pytest
import tifffile

from celestack.frame._metadata import (
    ExifMetadata,
    as_float,
    build_tiff_extratags,
    parse_celestack_metadata,
    parse_datetime_to_epoch,
    to_rational,
)

# --- as_float ---


def test_as_float_fraction_object() -> None:
    """Object with .num/.den attributes converts to float."""

    class FakeFraction:
        num = 30
        den = 1

    result = as_float(FakeFraction())
    assert result == pytest.approx(30.0)


def test_as_float_slash_string() -> None:
    """String with '/' parses as a fraction."""
    assert as_float("30/1") == pytest.approx(30.0)
    assert as_float("14/10") == pytest.approx(1.4)


def test_as_float_non_numeric() -> None:
    """Non-numeric string returns None."""
    assert as_float("not a number") is None


def test_as_float_none() -> None:
    """None input returns None."""
    assert as_float(None) is None


def test_as_float_plain_number() -> None:
    """Plain int and float convert directly."""
    assert as_float(42) == pytest.approx(42.0)
    assert as_float(3.14) == pytest.approx(3.14)


# --- parse_datetime_to_epoch ---


def test_parse_datetime_malformed() -> None:
    """Malformed datetime string returns None."""
    assert parse_datetime_to_epoch("not-a-date") is None


def test_parse_datetime_none() -> None:
    """None input returns None."""
    assert parse_datetime_to_epoch(None) is None


# --- to_rational ---


def test_to_rational_round_trip() -> None:
    """Float survives rational encoding within tolerance."""
    for value in [1.4, 30.0, 0.001, 24.0]:
        num, den = to_rational(value)
        recovered = num / den
        assert recovered == pytest.approx(value, rel=1e-6)


# --- parse_celestack_metadata ---


def test_parse_celestack_metadata_flat_json(tmp_path) -> None:
    """Flat JSON (backward compat) is parsed correctly."""
    meta = {"downscale_factor": 2, "timestamp": 12345.0}
    path = tmp_path / "flat.tif"
    arr = np.zeros((4, 4), dtype=np.uint8)
    tifffile.imwrite(path, arr, metadata=meta)

    with tifffile.TiffFile(path) as tif:
        page = cast(tifffile.TiffPage, tif.pages[0])
        result = parse_celestack_metadata(page)
    assert result.downscale_factor == 2
    assert result.timestamp == pytest.approx(12345.0)


def test_parse_celestack_metadata_invalid_json(tmp_path) -> None:
    """Invalid JSON in ImageDescription returns defaults."""
    path = tmp_path / "bad.tif"
    arr = np.zeros((4, 4), dtype=np.uint8)
    tifffile.imwrite(path, arr)

    # Overwrite ImageDescription with invalid JSON
    with tifffile.TiffFile(path) as tif:
        page = cast(tifffile.TiffPage, tif.pages[0])
        result = parse_celestack_metadata(page)
    assert result.downscale_factor == 1
    assert result.timestamp is None


# --- build_tiff_extratags ---


def test_build_tiff_extratags_empty() -> None:
    """All-None metadata produces an empty tag list."""
    meta = ExifMetadata()
    assert build_tiff_extratags(meta) == []
