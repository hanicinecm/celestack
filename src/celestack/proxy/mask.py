"""Downscaled boolean foreground mask stored as a NumPy ``.npy`` file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from celestack.config import CFG
from celestack.mask.core import Mask
from celestack.proxy._core import (
    load_downscale_factor,
    write_or_verify_downscale_factor,
)
from celestack.proxy._downscale import majority_vote
from celestack.proxy._plotting import plot_proxy_mask


class ProxyMask:
    """Boolean downscaled foreground mask backed by a ``.npy`` + sidecar TOML.

    Construction paths:

    - ``ProxyMask(path)``: load a boolean ``.npy`` array whose parent
      directory contains a ``metadata.toml`` sidecar declaring
      ``downscale_factor``.
    - ``ProxyMask.from_mask(mask, downscale_factor)``: create a detached
      in-memory proxy from a full-resolution :class:`~celestack.mask.Mask`
      via majority-vote downscaling.
    """

    def __init__(self, path: str | Path) -> None:
        """Load a boolean ``.npy`` proxy mask from disk.

        Args:
            path: Path to a ``.npy`` file containing a 2D boolean array.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the loaded array is not 2D boolean.
            ProxyMetadataError: If the sidecar ``metadata.toml`` is missing
                or malformed.
        """
        resolved = Path(path)
        if not resolved.exists():
            msg = f"ProxyMask not found: {resolved}"
            raise FileNotFoundError(msg)

        array = np.load(resolved)
        if array.ndim != 2 or array.dtype != np.bool_:
            msg = (
                f"ProxyMask requires a 2D boolean array; got shape {array.shape} "
                f"and dtype {array.dtype}"
            )
            raise ValueError(msg)

        self._path: Path | None = resolved
        self._array: np.ndarray = array
        self._downscale_factor: int = load_downscale_factor(resolved)

    @classmethod
    def from_mask(
        cls, mask: Mask, downscale_factor: int = CFG.proxy.default_downscale_factor
    ) -> ProxyMask:
        """Create a detached proxy from a full-resolution :class:`Mask`.

        Args:
            mask: Source full-resolution mask.
            downscale_factor: Integer downscale factor (>= 1).

        Returns:
            An in-memory :class:`ProxyMask` with no backing path.
        """
        instance = cls.__new__(cls)
        instance._path = None
        instance._array = majority_vote(mask.array, downscale_factor)
        instance._downscale_factor = int(downscale_factor)
        return instance

    @property
    def path(self) -> Path:
        """On-disk path to the ``.npy`` file.

        Raises:
            AttributeError: If the proxy is not backed by a file.
        """
        if self._path is None:
            msg = "ProxyMask is not backed by a file"
            raise AttributeError(msg)
        return self._path

    @property
    def shape(self) -> tuple[int, int]:
        """Proxy dimensions as ``(height, width)``."""
        return (int(self._array.shape[0]), int(self._array.shape[1]))

    @property
    def dtype(self) -> str:
        """Always ``"bool"``."""
        return "bool"

    @property
    def downscale_factor(self) -> int:
        """Integer scale factor relative to the source full-resolution mask."""
        return self._downscale_factor

    @property
    def array(self) -> np.ndarray:
        """Boolean 2D array of shape (H, W)."""
        return self._array

    def save_as(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Write the proxy mask to a ``.npy`` file and update the sidecar.

        Args:
            path: Destination ``.npy`` file path.
            overwrite: If False (default), raise if the destination exists.

        Raises:
            ValueError: If *path* does not have a ``.npy`` extension.
            FileExistsError: If the destination exists and *overwrite* is False.
            ProxyMetadataError: If an existing sidecar disagrees with this
                proxy's ``downscale_factor``.
        """
        target = Path(path)
        if target.suffix.lower() != ".npy":
            msg = f"Output path must be a .npy file, got: {target.suffix!r}"
            raise ValueError(msg)
        if target.exists() and not overwrite:
            msg = f"File already exists: {target}"
            raise FileExistsError(msg)
        target.parent.mkdir(parents=True, exist_ok=True)

        write_or_verify_downscale_factor(target, self._downscale_factor)
        np.save(target, self._array)
        self._path = target

    def plot(
        self,
        *,
        show_pixels: bool = True,
        upscale_coordinates: bool = True,
    ) -> go.Figure:
        """Render the proxy mask as a black-and-white Plotly figure.

        Args:
            show_pixels: If True (default), render as a Heatmap with
                per-pixel hover data. If False, render as a static PNG
                background image.
            upscale_coordinates: If True (default), scale the axes by
                the proxy's ``downscale_factor`` so hover coordinates
                are expressed in the full-resolution pixel space.

        Returns:
            A Plotly figure.
        """
        title = self._path.name if self._path is not None else "<detached>"
        return plot_proxy_mask(
            self._array,
            title=title,
            downscale_factor=self._downscale_factor,
            show_pixels=show_pixels,
            upscale_coordinates=upscale_coordinates,
        )

    def __repr__(self) -> str:
        """Return a concise debug representation of the proxy mask."""
        path_str = str(self._path) if self._path is not None else "<detached>"
        return f"ProxyMask({path_str!r}, downscale_factor={self._downscale_factor})"
