"""Downscaled background-subtracted float16 proxy frame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from celestack.config import CFG
from celestack.frame._image_ops import to_grayscale
from celestack.frame.core import Frame
from celestack.mask.core import Mask
from celestack.proxy._background import subtract_background
from celestack.proxy._core import (
    load_downscale_factor,
    write_or_verify_downscale_factor,
)
from celestack.proxy._downscale import block_average
from celestack.proxy._plotting import plot_proxy_frame
from celestack.proxy.mask import ProxyMask

_INT_DTYPES = frozenset({"uint8", "uint16"})


class ProxyFrame:
    """Downscaled, background-subtracted float16 proxy of a :class:`Frame`.

    The array is a 2D float16 image with sky noise centered near zero and
    foreground pixels set exactly to ``0.0``. Stars appear as positive
    peaks typically well below 1.0. Stored as a ``.npy`` file with a
    sidecar ``metadata.toml`` declaring ``downscale_factor``.
    """

    def __init__(self, path: str | Path) -> None:
        """Load a float16 proxy frame from a ``.npy`` file.

        Args:
            path: Path to a ``.npy`` file containing a 2D float16 array.

        Raises:
            FileNotFoundError: If *path* does not exist.
            ValueError: If the loaded array is not 2D float16.
            ProxyMetadataError: If the sidecar ``metadata.toml`` is missing
                or malformed.
        """
        resolved = Path(path)
        if not resolved.exists():
            msg = f"ProxyFrame not found: {resolved}"
            raise FileNotFoundError(msg)

        array = np.load(resolved)
        if array.ndim != 2 or array.dtype != np.float16:
            msg = (
                f"ProxyFrame requires a 2D float16 array; got shape {array.shape} "
                f"and dtype {array.dtype}"
            )
            raise ValueError(msg)

        self._path: Path | None = resolved
        self._array: np.ndarray = array
        self._downscale_factor: int = load_downscale_factor(resolved)

    @classmethod
    def from_frame(
        cls,
        frame: Frame,
        mask: Mask,
        downscale_factor: int,
        *,
        box_size: int | None = None,
        filter_size: int | None = None,
    ) -> ProxyFrame:
        """Build a detached proxy from a full-resolution frame and mask.

        Steps (order matters):

        1. Collapse the frame array to grayscale (preserves dtype).
        2. Block-average down by *downscale_factor*.
        3. Rescale into ``[0, 1]`` as float16 using the source integer
           dtype range.
        4. Build a :class:`ProxyMask` from *mask* at the same factor.
        5. Subtract a 2D sky background model (via photutils), zeroing
           masked pixels.

        Args:
            frame: Full-resolution source frame. Must be uint8 or uint16.
            mask: Full-resolution foreground mask matching *frame*.
            downscale_factor: Integer downscale factor (>= 1).
            box_size: Optional override for ``photutils.Background2D``
                box size. Defaults to ``CFG.proxy.background_box_size``.
            filter_size: Optional override for the ``Background2D`` median
                filter window. Defaults to ``CFG.proxy.background_filter_size``.

        Returns:
            A detached in-memory :class:`ProxyFrame`.

        Raises:
            ValueError: If *frame* has an unsupported dtype or the mask
                shape does not agree with the downscaled frame.
        """
        if frame.dtype not in _INT_DTYPES:
            msg = (
                f"ProxyFrame source must be uint8 or uint16; got dtype {frame.dtype!r}"
            )
            raise ValueError(msg)
        if mask.shape != frame.shape[:2]:
            msg = (
                f"Mask shape {mask.shape} does not match frame shape {frame.shape[:2]}"
            )
            raise ValueError(msg)

        box_size = CFG.proxy.background_box_size if box_size is None else box_size
        filter_size = (
            CFG.proxy.background_filter_size if filter_size is None else filter_size
        )

        gray = to_grayscale(frame.array)
        small = block_average(gray, downscale_factor)
        max_value = float(np.iinfo(np.dtype(frame.dtype)).max)
        rescaled = (small.astype(np.float32) / max_value).astype(np.float16)

        proxy_mask = ProxyMask.from_mask(mask, downscale_factor)
        if proxy_mask.shape != rescaled.shape:
            msg = (
                f"Downscaled mask shape {proxy_mask.shape} does not match "
                f"downscaled frame shape {rescaled.shape}"
            )
            raise ValueError(msg)

        result = subtract_background(
            rescaled,
            proxy_mask.array,
            box_size=box_size,
            filter_size=filter_size,
        )

        instance = cls.__new__(cls)
        instance._path = None
        instance._array = result
        instance._downscale_factor = int(downscale_factor)
        return instance

    @property
    def path(self) -> Path:
        """On-disk path to the ``.npy`` file.

        Raises:
            AttributeError: If the proxy is not backed by a file.
        """
        if self._path is None:
            msg = "ProxyFrame is not backed by a file"
            raise AttributeError(msg)
        return self._path

    @property
    def shape(self) -> tuple[int, int]:
        """Proxy dimensions as ``(height, width)``."""
        return (int(self._array.shape[0]), int(self._array.shape[1]))

    @property
    def dtype(self) -> str:
        """Always ``"float16"``."""
        return "float16"

    @property
    def downscale_factor(self) -> int:
        """Integer scale factor relative to the source full-resolution frame."""
        return self._downscale_factor

    @property
    def array(self) -> np.ndarray:
        """Float16 2D array of shape (H, W)."""
        return self._array

    def save_as(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Write the proxy frame to a ``.npy`` file and update the sidecar.

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

        np.save(target, self._array)
        write_or_verify_downscale_factor(target, self._downscale_factor)
        self._path = target

    def plot(self) -> go.Figure:
        """Render the proxy frame as a percentile-stretched grayscale image."""
        title = self._path.name if self._path is not None else "<detached>"
        return plot_proxy_frame(self._array, title=title)

    def __repr__(self) -> str:
        """Return a concise debug representation of the proxy frame."""
        path_str = str(self._path) if self._path is not None else "<detached>"
        return f"ProxyFrame({path_str!r}, downscale_factor={self._downscale_factor})"
