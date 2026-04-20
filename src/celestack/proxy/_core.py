"""Shared helpers for proxy sidecar metadata I/O."""

from __future__ import annotations

import tomllib
from pathlib import Path

from celestack.exceptions import ProxyMetadataError

METADATA_FILENAME = "metadata.toml"
_DOWNSCALE_KEY = "downscale_factor"


def metadata_path_for(npy_path: Path) -> Path:
    """Return the sidecar metadata path for a proxy ``.npy`` file."""
    return npy_path.parent / METADATA_FILENAME


def load_downscale_factor(npy_path: Path) -> int:
    """Read ``downscale_factor`` from the sidecar metadata for *npy_path*.

    Args:
        npy_path: Path to a proxy ``.npy`` file. Its parent directory must
            contain a ``metadata.toml`` sidecar.

    Returns:
        The downscale factor recorded in the sidecar.

    Raises:
        ProxyMetadataError: If the sidecar is missing, malformed, or the
            ``downscale_factor`` key is absent or non-integer.
    """
    meta_path = metadata_path_for(npy_path)
    if not meta_path.exists():
        msg = f"Proxy metadata sidecar missing: {meta_path}"
        raise ProxyMetadataError(msg)

    try:
        with meta_path.open("rb") as fp:
            data = tomllib.load(fp)
    except tomllib.TOMLDecodeError as exc:
        msg = f"Malformed proxy metadata TOML: {meta_path} ({exc})"
        raise ProxyMetadataError(msg) from exc

    if _DOWNSCALE_KEY not in data:
        msg = f"Missing key {_DOWNSCALE_KEY!r} in proxy metadata: {meta_path}"
        raise ProxyMetadataError(msg)

    value = data[_DOWNSCALE_KEY]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        msg = (
            f"Invalid {_DOWNSCALE_KEY} in proxy metadata {meta_path}: "
            f"expected positive integer, got {value!r}"
        )
        raise ProxyMetadataError(msg)

    return value


def write_or_verify_downscale_factor(npy_path: Path, downscale_factor: int) -> None:
    """Write the sidecar metadata or verify an existing one agrees.

    If the sidecar does not exist, it is created with
    ``downscale_factor = <value>``. If it exists, the recorded value must
    match *downscale_factor*.

    Args:
        npy_path: Path to the proxy ``.npy`` file.
        downscale_factor: Factor to record or verify.

    Raises:
        ProxyMetadataError: If an existing sidecar disagrees with the
            supplied factor (or is malformed).
    """
    meta_path = metadata_path_for(npy_path)
    if meta_path.exists():
        existing = load_downscale_factor(npy_path)
        if existing != downscale_factor:
            msg = (
                f"Proxy metadata mismatch at {meta_path}: "
                f"existing downscale_factor={existing}, new={downscale_factor}"
            )
            raise ProxyMetadataError(msg)
        return

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(f"{_DOWNSCALE_KEY} = {int(downscale_factor)}\n")
