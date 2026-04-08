"""Algorithmic mask builder using K-Means clustering."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from celestack.exceptions import MaskError
from celestack.frame.core import Frame
from celestack.mask._clustering import (
    ClusterWeights,
    _normalize_rgb,
    extract_features,
    run_kmeans,
)
from celestack.mask.core import Mask
from celestack.mask._plotting import plot_clusters as _plot_clusters
from celestack.mask._plotting import plot_mask as _plot_mask


class MaskBuilder:
    """Algorithmic builder that creates a foreground mask via K-Means clustering.

    Instantiate with an RGB source image.  All clustering and plotting
    operates on an internal RGB proxy (downscaled copy) for speed.  Call
    ``compute_clusters``, ``apply_labels``, inspect via ``plot_clusters``
    / ``plot_mask``, and finally ``build()`` to re-run clustering on the
    full-resolution image and obtain a finished :class:`Mask`.
    """

    def __init__(
        self,
        image: Frame,
        *,
        proxy_downscale_factor: int = 4,
        proxy_bit_depth: int = 8,
    ) -> None:
        """Initialize with an RGB image for algorithmic mask building.

        A downscaled RGB proxy is created immediately and used for all
        interactive clustering and plotting.

        Args:
            image: Source RGB frame to cluster.  Must be 3-channel.
            proxy_downscale_factor: Factor by which to downscale the source
                image for the interactive proxy.  Must be >= 1.
            proxy_bit_depth: Bit depth of the proxy image (default 8).

        Raises:
            MaskError: If the image is not 3-channel RGB.
        """
        if len(image.shape) != 3 or image.shape[2] < 3:
            msg = "MaskBuilder requires a 3-channel RGB image"
            raise MaskError(msg)

        # The original image:
        self._image: Frame = image
        self._height: int = image.shape[0]
        self._width: int = image.shape[1]

        # The downscaled RGB proxy used for clustering and plotting:
        self._proxy: Frame = image.downscaled_copy(
            proxy_downscale_factor,
            bit_depth=proxy_bit_depth,
            grayscale=False,
        )
        self._proxy_height: int = self._proxy.shape[0]
        self._proxy_width: int = self._proxy.shape[1]
        self._proxy_downscale_factor: int = proxy_downscale_factor

        self._features: np.ndarray | None = None
        self._current_weights: ClusterWeights | None = None
        self._labels: np.ndarray | None = None
        self._n_clusters: int | None = None
        self._foreground_clusters: set[int] | None = None
        self._mask: np.ndarray | None = None

    def compute_clusters(
        self,
        n_clusters: int,
        weights: ClusterWeights | None = None,
    ) -> None:
        """Run K-Means clustering on the proxy image.

        Can be called multiple times with different values of K or
        different weights.  The feature matrix is recomputed only when
        ``weights`` changes.

        Args:
            n_clusters: Number of clusters for K-Means.
            weights: Per-channel feature weights.  Defaults to
                ``ClusterWeights()`` (RGB only, no spatial features).

        Raises:
            ValueError: If n_clusters < 2.
        """
        if weights is None:
            weights = ClusterWeights()

        if n_clusters < 2:
            msg = "n_clusters must be at least 2"
            raise ValueError(msg)

        proxy_arr = self._proxy.array
        if self._features is None or weights != self._current_weights:
            self._features = extract_features(
                proxy_arr, self._proxy.bit_depth, weights
            )
            self._current_weights = weights
        rgb_norm = _normalize_rgb(proxy_arr, self._proxy.bit_depth)

        self._labels = run_kmeans(
            self._features,
            n_clusters,
            self._proxy_height,
            self._proxy_width,
            rgb_norm=rgb_norm,
        )
        self._n_clusters = n_clusters
        self._foreground_clusters = None
        self._mask = None

    def apply_labels(self, foreground_clusters: set[int]) -> None:
        """Designate which cluster labels are foreground.

        Builds the boolean mask (at proxy resolution) from cluster
        assignments.

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
        self._mask = np.isin(self._labels, list(foreground_clusters))

    def build(self) -> Mask:
        """Re-run clustering on the full-resolution image and return the mask.

        Uses the same weights and foreground cluster selection as the
        proxy session.  The result is a detached in-memory :class:`Mask`
        at full resolution.

        Returns:
            A full-resolution Mask wrapping the clustered boolean array.

        Raises:
            MaskError: If no mask has been created yet. Call
                ``apply_labels`` first.
        """
        if (
            self._foreground_clusters is None
            or self._n_clusters is None
            or self._current_weights is None
        ):
            msg = "No mask available; call apply_labels first"
            raise MaskError(msg)

        with self._image._conserve_cache():
            arr = self._image.array
            rgb_norm = _normalize_rgb(arr, self._image.bit_depth)
            features = extract_features(
                arr, self._image.bit_depth, self._current_weights
            )
            labels = run_kmeans(
                features,
                self._n_clusters,
                self._height,
                self._width,
                rgb_norm=rgb_norm,
            )

        full_mask = np.isin(labels, list(self._foreground_clusters))
        return Mask._from_array(full_mask)

    def plot_clusters(self) -> go.Figure:
        """Visualize cluster labels on the proxy image as a color-coded plot.

        Returns:
            Plotly figure with color-coded cluster regions and a legend.

        Raises:
            MaskError: If ``compute_clusters`` has not been called yet.
        """
        if self._labels is None:
            msg = "compute_clusters must be called before plot_clusters"
            raise MaskError(msg)

        return _plot_clusters(self._labels)

    def plot_mask(self) -> go.Figure:
        """Visualize the current proxy mask as a black-and-white plot.

        Returns:
            Plotly figure with the mask rendered as a black-and-white image.

        Raises:
            MaskError: If no mask has been created yet.
        """
        if self._mask is None:
            msg = "A mask must exist before calling plot_mask"
            raise MaskError(msg)

        return _plot_mask(self._mask)

    @property
    def mask_array(self) -> np.ndarray:
        """Current boolean foreground mask at proxy resolution with shape (H, W).

        Raises:
            MaskError: If no mask has been created yet. Call
                ``apply_labels`` first.
        """
        if self._mask is None:
            msg = "No mask available; call apply_labels first"
            raise MaskError(msg)
        return self._mask
