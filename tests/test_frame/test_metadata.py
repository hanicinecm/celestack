"""Tests for frame metadata parsing and serialization."""

import pytest

from celestack.frame._metadata import (
    ExifMetadata,
    as_float,
    build_tiff_extratags,
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


# --- build_tiff_extratags ---


def test_build_tiff_extratags_empty() -> None:
    """All-None metadata produces an empty tag list."""
    meta = ExifMetadata()
    assert build_tiff_extratags(meta) == []
