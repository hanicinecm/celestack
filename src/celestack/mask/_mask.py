"""Standalone boolean mask container."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import tifffile

from celestack.exceptions import DownscaleError
from celestack.frame._backends import get_backends, is_tiff_path
from celestack.frame._constants import CELESTACK_KEY
from celestack.frame._image_ops import downscale_by_block_average, to_grayscale


class Mask:
    """Boolean foreground mask backed by an 8-bit grayscale TIFF.

    Construction paths:

    - ``Mask(path)``: load a mask from any supported image file.
    - ``Mask._from_array(array)``: create an in-memory mask from a boolean
      NumPy array (used internally by :class:`MaskBuilder`).

    ``array`` always returns a 2D ``bool`` array regardless of the on-disk
    format. Grayscale and RGB inputs are booleanized on first load (non-zero
    pixels become ``True``). The booleanized array is cached; ``unload()``
    drops it to free memory.
    """

    def __init__(self, path: str | Path) -> None:
        """Load a mask from an image file.

        Args:
            path: Source image path. TIFF, JPEG, and PNG are supported.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the image format is unsupported.
        """
        resolved = Path(path)
        if not resolved.exists():
            msg = f"Mask not found: {resolved}"
            raise FileNotFoundError(msg)

        self._path: Path | None = resolved
        backends = get_backends()
        suffix = resolved.suffix.lower()
        if suffix not in backends:
            msg = f"No backend found for: {resolved.suffix}"
            raise ValueError(msg)
        self._backend = backends[suffix]
        info = self._backend.inspect(resolved)
        self._shape: tuple[int, int] = (info.shape[0], info.shape[1])
        self._downscale_factor: int = info.celestack_metadata.downscale_factor
        self._array_cache: np.ndarray | None = None

    @classmethod
    def _from_array(cls, array: np.ndarray, *, downscale_factor: int = 1) -> Mask:
        """Create an in-memory mask from a boolean array.

        The array is cast to ``bool`` if it is not already. The resulting
        mask has no backing path and a ``downscale_factor`` of 1 by default,
        but this can be overridden.

        Args:
            array: 2D array of shape (H, W).
            downscale_factor: Linear downscale factor relative to the full-resolution
                source.

        Returns:
            A Mask with no backing path.
        """
        instance = cls.__new__(cls)
        instance._path = None
        instance._backend = None
        instance._shape = (int(array.shape[0]), int(array.shape[1]))
        instance._downscale_factor = downscale_factor
        instance._array_cache = np.array(array, dtype=np.bool_)
        return instance

    @property
    def path(self) -> Path:
        """Return the on-disk path represented by this mask.

        Raises:
            AttributeError: If the mask is not backed by a file.
        """
        if self._path is None:
            msg = "Mask is not backed by a file"
            raise AttributeError(msg)
        return self._path

    @property
    def shape(self) -> tuple[int, int]:
        """Mask dimensions as ``(height, width)``."""
        return self._shape

    @property
    def downscale_factor(self) -> int:
        """Linear downscale factor relative to the full-resolution source.

        Defaults to 1 for full-resolution and external masks. Proxy masks
        written by :meth:`save_as` on a downscaled copy embed their factor
        in Celestack metadata, which is read back on construction.
        """
        return self._downscale_factor

    @property
    def array(self) -> np.ndarray:
        """Boolean mask array of shape (H, W), loaded lazily on first access.

        Raises:
            ValueError: If the mask is in-memory with no backing path and
                no cached array (should not occur in normal usage).
        """
        if self._array_cache is None:
            if self._path is None:
                msg = "In-memory mask has no backing path and no cached array"
                raise ValueError(msg)
            raw = self._backend.load_array(self._path)
            if raw.dtype == np.bool_:
                self._array_cache = raw.copy()
            elif raw.ndim == 3:
                gray = to_grayscale(raw)
                self._array_cache = gray != 0
            else:
                self._array_cache = raw != 0

        if self._array_cache is None:
            msg = f"Failed to load mask array from path: {self._path}"
            raise ValueError(msg)

        return self._array_cache

    def unload(self) -> None:
        """Drop the cached boolean array to free memory."""
        self._array_cache = None

    @contextmanager
    def _conserve_cache(self) -> Iterator[None]:
        """Restore the array cache to its prior state on exit.

        If the array was not loaded before entering the block, it is
        unloaded when the block exits.
        """
        was_loaded = self._array_cache is not None
        yield
        if not was_loaded:
            self.unload()

    def save_as(self, path: str | Path, *, overwrite: bool = False) -> None:
        """Write the mask to an 8-bit grayscale TIFF and bind this instance to that path.

        Always writes from the boolean array (``True`` → 255, ``False`` → 0).
        Never copies an existing file verbatim, ensuring the on-disk format
        is always a clean 8-bit grayscale TIFF. Proxy masks
        (``downscale_factor > 1``) have Celestack metadata embedded automatically.

        The array cache is conserved: if it was not loaded before the call,
        it is unloaded afterwards.

        Args:
            path: Destination file path (.tif or .tiff).
            overwrite: If False (default), raise if the destination exists.

        Raises:
            ValueError: If *path* does not have a TIFF extension.
            FileExistsError: If the destination exists and *overwrite* is False.
        """
        target = Path(path)
        if not is_tiff_path(target):
            msg = f"Output path must be a TIFF file, got: {target.suffix!r}"
            raise ValueError(msg)
        if target.exists() and not overwrite:
            msg = f"File already exists: {target}"
            raise FileExistsError(msg)
        target.parent.mkdir(parents=True, exist_ok=True)

        if self._downscale_factor > 1:
            metadata: dict | None = {
                CELESTACK_KEY: {"downscale_factor": int(self._downscale_factor)}
            }
        else:
            metadata = None

        with self._conserve_cache():
            data = self.array.astype(np.uint8) * 255
            tifffile.imwrite(
                target,
                data=data,
                compression="zlib",
                photometric="minisblack",
                metadata=metadata,
            )

        self._path = target
        backends = get_backends()
        self._backend = backends[target.suffix.lower()]

    def downscaled_copy(self, downscale_factor: int) -> Mask:
        """Return a detached downscaled copy of this mask.

        Uses majority voting: blocks where more than half the pixels are
        foreground become foreground in the downscaled result. The result
        is held in memory. Call :meth:`save_as` on the returned mask to
        persist it; the ``downscale_factor`` is embedded in Celestack
        metadata automatically.

        The array cache is conserved: if it was not loaded before the call,
        it is unloaded afterwards.

        Args:
            downscale_factor: Integer linear downscale factor (>= 1).

        Returns:
            A new detached Mask with the downscaled boolean data.

        Raises:
            DownscaleError: If called on an already-downscaled mask.
            ValueError: If *downscale_factor* is less than 1.
        """
        if self._downscale_factor != 1:
            msg = "Cannot downscale a proxy mask"
            raise DownscaleError(msg)
        if downscale_factor < 1:
            msg = "downscale_factor must be >= 1"
            raise ValueError(msg)

        with self._conserve_cache():
            averaged = downscale_by_block_average(self.array, downscale_factor)
            downscaled = averaged > 0.5

        return Mask._from_array(downscaled, downscale_factor=downscale_factor)

    def __repr__(self) -> str:
        """Return a concise debug representation of the mask."""
        path_str = str(self._path) if self._path is not None else "<detached>"
        return f"Mask({path_str!r}, downscale_factor={self._downscale_factor})"
