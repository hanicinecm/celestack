"""Algorithmic mask builder using K-Means clustering."""

from __future__ import annotations

from dataclasses import astuple

import numpy as np
import plotly.graph_objects as go

from celestack.exceptions import MaskError
from celestack.frame.core import Frame
from celestack.mask._clustering import (
    ClusterWeights,
    extract_features,
    relabel_by_descending_y,
    run_kmeans,
)
from celestack.mask._plotting import plot_clusters as _plot_clusters
from celestack.mask.core import Mask


class MaskBuilder:
    """Algorithmic builder that creates a foreground mask via K-Means clustering.

    Instantiate with an RGB source image.  Standardized features are
    extracted once at construction time.  Call ``compute_clusters`` (which
    applies weights and runs K-Means), inspect via ``plot_clusters``, and
    finally ``build(foreground_clusters)`` to obtain a finished :class:`Mask`.
    """

    def __init__(self, image: Frame) -> None:
        """Initialize with an RGB image for algorithmic mask building.

        Extracts and standardizes the full-resolution feature matrix
        (R, G, B, X, Y) immediately.

        Args:
            image: Source RGB frame to cluster.  Must be 3-channel.

        Raises:
            MaskError: If the image is not 3-channel RGB.
        """
        if len(image.shape) != 3 or image.shape[2] < 3:
            msg = "MaskBuilder requires a 3-channel RGB image"
            raise MaskError(msg)

        self._height: int = image.shape[0]
        self._width: int = image.shape[1]
        self._std_features: np.ndarray = extract_features(image.array)

        self._labels: np.ndarray | None = None
        self._n_clusters: int | None = None

    def compute_clusters(
        self,
        n_clusters: int,
        weights: ClusterWeights | None = None,
    ) -> None:
        """Run K-Means clustering on the source image.

        Applies the given weights to the stored standardized features,
        drops any zero-weighted columns, and runs K-Means.  Can be called
        multiple times with different K or weights.

        Args:
            n_clusters: Number of clusters for K-Means.
            weights: Per-channel feature weights.  Defaults to
                ``ClusterWeights()`` (RGB only, no spatial features).

        Raises:
            ValueError: If n_clusters < 2 or all weights are zero.
        """
        if weights is None:
            weights = ClusterWeights()

        if n_clusters < 2:
            msg = "n_clusters must be at least 2"
            raise ValueError(msg)

        weight_array = np.array(astuple(weights), dtype=np.float32)
        nonzero = weight_array != 0.0
        if not nonzero.any():
            msg = "At least one weight must be non-zero"
            raise ValueError(msg)

        weighted_features = self._std_features[:, nonzero] * weight_array[nonzero]

        flat_labels = run_kmeans(weighted_features, n_clusters)
        labels_2d = flat_labels.reshape(self._height, self._width)
        self._labels = relabel_by_descending_y(labels_2d, n_clusters)
        self._n_clusters = n_clusters

    def build(self, foreground_clusters: set[int]) -> Mask:
        """Build a boolean mask from the current cluster labels.

        Args:
            foreground_clusters: Set of cluster indices to mark as
                foreground.

        Returns:
            A detached in-memory full-resolution Mask.

        Raises:
            MaskError: If ``compute_clusters`` has not been called yet.
            ValueError: If any cluster index is out of range.
        """
        if self._labels is None or self._n_clusters is None:
            msg = "compute_clusters must be called before build"
            raise MaskError(msg)

        invalid = {c for c in foreground_clusters if not (0 <= c < self._n_clusters)}
        if invalid:
            msg = (
                f"Cluster indices out of range [0, {self._n_clusters}): "
                f"{sorted(invalid)}"
            )
            raise ValueError(msg)

        mask_array = np.isin(self._labels, list(foreground_clusters))
        return Mask._from_array(mask_array)

    def plot_clusters(self) -> go.Figure:
        """Visualize cluster labels as a color-coded plot.

        Returns:
            Plotly figure with color-coded cluster regions and a legend.

        Raises:
            MaskError: If ``compute_clusters`` has not been called yet.
        """
        if self._labels is None:
            msg = "compute_clusters must be called before plot_clusters"
            raise MaskError(msg)

        return _plot_clusters(self._labels)
