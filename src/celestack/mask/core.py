"""Public MaskBuilder implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

from celestack.exceptions import MaskError
from celestack.frame.core import Frame
from celestack.mask._clustering import extract_features, run_kmeans
from celestack.mask._mask import Mask
from celestack.mask._plotting import plot_clusters as _plot_clusters
from celestack.mask._plotting import plot_mask as _plot_mask


@dataclass
class _RectEdit:
    x0: int
    y0: int
    x1: int
    y1: int
    foreground: bool


@dataclass
class _PixelEdit:
    pixels: np.ndarray
    foreground: bool


_Edit = _RectEdit | _PixelEdit


class MaskBuilder:
    """Stateful builder for foreground/background masks.

    Instantiate with an RGB source image, call ``compute_clusters``,
    ``apply_labels``, and optionally refine with ``mask_rectangle`` or
    ``mask_pixels``. Call ``build()`` to obtain a finished :class:`Mask`.

    Manual edits are recorded and replayed after any re-clustering or
    re-labeling, so adjustments are preserved across workflow iterations.
    """

    def __init__(self, image: Frame) -> None:
        """Initialize with an RGB image for algorithmic mask building.

        Args:
            image: Source RGB frame to cluster. Must be 3-channel.

        Raises:
            MaskError: If the image is not 3-channel RGB.
        """
        if len(image.shape) != 3 or image.shape[2] < 3:
            msg = "MaskBuilder requires a 3-channel RGB image"
            raise MaskError(msg)

        self._image: Frame = image
        self._bit_depth: int = image.bit_depth
        self._height: int = image.shape[0]
        self._width: int = image.shape[1]
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._n_clusters: int | None = None
        self._foreground_clusters: set[int] | None = None
        self._mask: np.ndarray | None = None
        self._edits: list[_Edit] = []

    def compute_clusters(self, n_clusters: int) -> None:
        """Run K-Means clustering on the image.

        Can be called multiple times with different values of K.
        Manual edits are preserved and will be replayed when
        ``apply_labels`` is next called.

        Args:
            n_clusters: Number of clusters for K-Means.

        Raises:
            ValueError: If n_clusters < 2.
        """
        if n_clusters < 2:
            msg = "n_clusters must be at least 2"
            raise ValueError(msg)

        if self._features is None:
            self._features = extract_features(self._image.array, self._bit_depth)

        self._labels = run_kmeans(self._features, n_clusters, self._height, self._width)
        self._n_clusters = n_clusters
        self._foreground_clusters = None
        self._mask = None

    def apply_labels(self, foreground_clusters: set[int]) -> None:
        """Designate which cluster labels are foreground.

        Builds the boolean mask from cluster assignments, then replays
        any manual edits in order.

        Args:
            foreground_clusters: Set of cluster indices to mark as
                foreground.

        Raises:
            MaskError: If ``compute_clusters`` has not been called yet.
            ValueError: If any cluster index is out of range.
        """
        if self._labels is None:
            msg = "compute_clusters must be called before apply_labels"
            raise MaskError(msg)
        if self._n_clusters is None:
            msg = "Internal error: _labels exists but _n_clusters is None"
            raise MaskError(msg)

        invalid = {c for c in foreground_clusters if not (0 <= c < self._n_clusters)}
        if invalid:
            msg = (
                f"Cluster indices out of range [0, {self._n_clusters}): "
                f"{sorted(invalid)}"
            )
            raise ValueError(msg)

        self._foreground_clusters = foreground_clusters
        base_mask = np.isin(self._labels, list(foreground_clusters))
        self._replay_edits(base_mask)
        self._mask = base_mask

    def mask_rectangle(
        self, x0: int, y0: int, x1: int, y1: int, *, foreground: bool
    ) -> None:
        """Manually set a rectangular region of the mask.

        The edit is recorded and replayed after any future re-clustering
        or re-labeling.

        Args:
            x0: Left column (inclusive).
            y0: Top row (inclusive).
            x1: Right column (exclusive).
            y1: Bottom row (exclusive).
            foreground: Whether to mark the region as foreground.

        Raises:
            MaskError: If no mask has been created yet.
            ValueError: If coordinates are out of bounds or inverted.
        """
        if self._mask is None:
            msg = "A mask must exist before applying rectangle edits"
            raise MaskError(msg)

        self._validate_rect(x0, y0, x1, y1)
        self._edits.append(_RectEdit(x0=x0, y0=y0, x1=x1, y1=y1, foreground=foreground))
        self._mask[y0:y1, x0:x1] = foreground

    def mask_pixels(self, pixels: np.ndarray, *, foreground: bool) -> None:
        """Manually set specific pixels in the mask.

        Args:
            pixels: Array of shape (N, 2) with (x, y) coordinates, where
                x is the column index and y is the row index.
            foreground: Whether to mark the pixels as foreground.

        Raises:
            MaskError: If no mask has been created yet.
            ValueError: If pixels has an unexpected shape or any
                coordinates are out of bounds.
        """
        if self._mask is None:
            msg = "A mask must exist before applying pixel edits"
            raise MaskError(msg)

        pixels = np.asarray(pixels)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            msg = "pixels must have shape (N, 2)"
            raise ValueError(msg)

        if pixels.size > 0:
            xs, ys = pixels[:, 0], pixels[:, 1]
            if (
                xs.min() < 0
                or ys.min() < 0
                or xs.max() >= self._width
                or ys.max() >= self._height
            ):
                msg = "Pixel coordinates out of image bounds"
                raise ValueError(msg)

        self._edits.append(_PixelEdit(pixels=pixels.copy(), foreground=foreground))
        self._mask[pixels[:, 1], pixels[:, 0]] = foreground

    def build(self) -> Mask:
        """Return the current mask as a finished in-memory :class:`Mask`.

        Returns:
            A Mask wrapping the current boolean array.

        Raises:
            MaskError: If no mask has been created yet. Call
                ``apply_labels`` first.
        """
        return Mask._from_array(self.mask_array)

    def plot_clusters(self) -> go.Figure:
        """Visualize cluster labels overlaid on the source image.

        Returns:
            Plotly figure with color-coded cluster overlay.

        Raises:
            MaskError: If ``compute_clusters`` has not been called yet.
        """
        if self._labels is None:
            msg = "compute_clusters must be called before plot_clusters"
            raise MaskError(msg)

        with self._image._conserve_cache():
            return _plot_clusters(self._image.array, self._labels, self._bit_depth)

    def plot_mask(self) -> go.Figure:
        """Visualize the current mask overlaid on the source image.

        Returns:
            Plotly figure with semi-transparent mask overlay.

        Raises:
            MaskError: If no mask has been created yet.
        """
        if self._mask is None:
            msg = "A mask must exist before calling plot_mask"
            raise MaskError(msg)

        with self._image._conserve_cache():
            return _plot_mask(self._image.array, self._mask, self._bit_depth)

    @property
    def mask_array(self) -> np.ndarray:
        """Current boolean foreground mask array with shape (H, W).

        Raises:
            MaskError: If no mask has been created yet. Call
                ``apply_labels`` first.
        """
        if self._mask is None:
            msg = "No mask available; call apply_labels first"
            raise MaskError(msg)
        return self._mask

    def _validate_rect(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """Validate rectangle coordinates against image bounds.

        Raises:
            ValueError: If coordinates are inverted or outside bounds.
        """
        if x1 <= x0 or y1 <= y0:
            msg = "x1 must be greater than x0 and y1 must be greater than y0"
            raise ValueError(msg)
        if x0 < 0 or y0 < 0:
            msg = "x0 and y0 must be non-negative"
            raise ValueError(msg)
        if x1 > self._width or y1 > self._height:
            msg = "Rectangle extends outside image bounds"
            raise ValueError(msg)

    def _replay_edits(self, mask: np.ndarray) -> None:
        """Apply all recorded manual edits to a mask array in place.

        Args:
            mask: Boolean mask array to modify.
        """
        for edit in self._edits:
            if isinstance(edit, _RectEdit):
                mask[edit.y0 : edit.y1, edit.x0 : edit.x1] = edit.foreground
            elif isinstance(edit, _PixelEdit):
                mask[edit.pixels[:, 1], edit.pixels[:, 0]] = edit.foreground
