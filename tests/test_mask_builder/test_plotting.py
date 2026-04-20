"""Tests for mask_builder plotting helpers."""

import numpy as np
import plotly.graph_objects as go
import pytest

from celestack.exceptions import MaskError
from celestack.frame.core import Frame
from celestack.mask_builder._plotting import plot_clusters
from celestack.mask_builder.core import MaskBuilder


def test_plot_clusters_returns_figure() -> None:
    """plot_clusters returns a Plotly Figure."""
    labels = np.zeros((12, 16), dtype=np.int32)
    labels[:, 8:] = 1
    fig = plot_clusters(labels)
    assert isinstance(fig, go.Figure)


def test_plot_clusters_has_layout_image() -> None:
    """plot_clusters adds the cluster image as a layout image."""
    labels = np.zeros((8, 10), dtype=np.int32)
    fig = plot_clusters(labels)
    assert len(fig.layout.images) > 0


def test_plot_clusters_has_no_image_trace() -> None:
    """plot_clusters does not use go.Image traces."""
    labels = np.zeros((8, 10), dtype=np.int32)
    fig = plot_clusters(labels)
    assert not any(isinstance(t, go.Image) for t in fig.data)


def test_plot_clusters_has_scatter_traces_for_legend() -> None:
    """plot_clusters adds one invisible Scatter trace per cluster."""
    n_clusters = 3
    labels = (np.arange(8 * 10) % n_clusters).reshape(8, 10).astype(np.int32)
    fig = plot_clusters(labels)
    scatter_traces = [t for t in fig.data if isinstance(t, go.Scatter)]
    assert len(scatter_traces) == n_clusters


def test_plot_clusters_legend_not_interactive() -> None:
    """Legend itemclick and itemdoubleclick are disabled."""
    labels = np.zeros((6, 8), dtype=np.int32)
    fig = plot_clusters(labels)
    assert fig.layout.legend.itemclick is False
    assert fig.layout.legend.itemdoubleclick is False


@pytest.mark.parametrize("n_clusters", [2, 5])
def test_plot_clusters_title_contains_k(n_clusters: int) -> None:
    """Figure title contains the number of clusters."""
    labels = (np.arange(6 * 8) % n_clusters).reshape(6, 8).astype(np.int32)
    fig = plot_clusters(labels)
    assert str(n_clusters) in str(fig.layout.title.text)


def test_plot_clusters_legend_labels_match_cluster_ids() -> None:
    """Each Scatter trace is named 'Cluster N' for N in [0, K)."""
    n_clusters = 4
    labels = (np.arange(8 * 10) % n_clusters).reshape(8, 10).astype(np.int32)
    fig = plot_clusters(labels)
    names = {t.name for t in fig.data if isinstance(t, go.Scatter)}
    assert names == {f"Cluster {i}" for i in range(n_clusters)}


def test_mb_plot_clusters_returns_figure(builder: MaskBuilder) -> None:
    """MaskBuilder.plot_clusters returns a Figure after clustering."""
    assert isinstance(builder.plot_clusters(), go.Figure)


def test_mb_plot_clusters_before_clustering_raises(rgb_frame: Frame) -> None:
    """MaskBuilder.plot_clusters raises MaskError before clustering."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(MaskError, match="compute_clusters"):
        mb.plot_clusters()
