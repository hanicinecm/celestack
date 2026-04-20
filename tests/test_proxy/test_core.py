"""Tests for proxy sidecar metadata.toml helpers."""

from pathlib import Path

import pytest

from celestack.exceptions import ProxyMetadataError
from celestack.proxy._core import (
    METADATA_FILENAME,
    load_downscale_factor,
    metadata_path_for,
    write_or_verify_downscale_factor,
)


def test_metadata_path_for_returns_sibling(tmp_path: Path) -> None:
    """Sidecar path sits next to the .npy file."""
    npy = tmp_path / "sub" / "p.npy"
    assert metadata_path_for(npy) == tmp_path / "sub" / METADATA_FILENAME


def test_write_and_load_roundtrip(tmp_path: Path) -> None:
    """Writing a sidecar then reading it returns the same factor."""
    npy = tmp_path / "p.npy"
    write_or_verify_downscale_factor(npy, 4)
    assert load_downscale_factor(npy) == 4


def test_write_verifies_matching_value(tmp_path: Path) -> None:
    """Writing the same factor twice is idempotent."""
    npy = tmp_path / "p.npy"
    write_or_verify_downscale_factor(npy, 2)
    write_or_verify_downscale_factor(npy, 2)
    assert load_downscale_factor(npy) == 2


def test_write_rejects_mismatch(tmp_path: Path) -> None:
    """Existing sidecar with a different factor raises."""
    npy = tmp_path / "p.npy"
    write_or_verify_downscale_factor(npy, 2)
    with pytest.raises(ProxyMetadataError, match="mismatch"):
        write_or_verify_downscale_factor(npy, 4)


def test_load_missing_sidecar_raises(tmp_path: Path) -> None:
    """Missing sidecar raises ProxyMetadataError."""
    with pytest.raises(ProxyMetadataError, match="missing"):
        load_downscale_factor(tmp_path / "p.npy")


def test_load_malformed_toml_raises(tmp_path: Path) -> None:
    """Malformed TOML raises ProxyMetadataError."""
    (tmp_path / METADATA_FILENAME).write_text("not = = valid\n")
    with pytest.raises(ProxyMetadataError, match="Malformed"):
        load_downscale_factor(tmp_path / "p.npy")


def test_load_missing_key_raises(tmp_path: Path) -> None:
    """Missing downscale_factor key raises ProxyMetadataError."""
    (tmp_path / METADATA_FILENAME).write_text("other = 1\n")
    with pytest.raises(ProxyMetadataError, match="Missing key"):
        load_downscale_factor(tmp_path / "p.npy")


@pytest.mark.parametrize("value", ["4", 0, -1, 1.5, True])
def test_load_invalid_value_raises(tmp_path: Path, value: object) -> None:
    """Non-positive-integer values are rejected."""
    (tmp_path / METADATA_FILENAME).write_text(
        f"downscale_factor = {value!r}\n"
        if isinstance(value, str)
        else f"downscale_factor = {value}\n"
    )
    with pytest.raises(ProxyMetadataError, match="Invalid"):
        load_downscale_factor(tmp_path / "p.npy")
