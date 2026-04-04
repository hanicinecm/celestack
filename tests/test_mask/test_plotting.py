"""Tests for mask plotting helpers."""

import numpy as np
import plotly.graph_objects as go
import pytest

from celestack.exceptions import MaskError
from celestack.frame.core import Frame
from celestack.mask._plotting import plot_clusters, plot_mask
from celestack.mask.core import MaskBuilder

# ---------------------------------------------------------------------------
# plot_clusters (module-level function)
# ---------------------------------------------------------------------------


def test_plot_clusters_returns_figure() -> None:
    """plot_clusters returns a Plotly Figure."""
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    labels = np.zeros((12, 16), dtype=np.int32)
    labels[:, 8:] = 1
    fig = plot_clusters(image, labels, bit_depth=8)
    assert isinstance(fig, go.Figure)


def test_plot_clusters_has_overlay_trace() -> None:
    """plot_clusters includes a go.Image trace for the color overlay."""
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    labels = np.zeros((8, 10), dtype=np.int32)
    fig = plot_clusters(image, labels, bit_depth=8)
    assert any(isinstance(t, go.Image) for t in fig.data)


def test_plot_clusters_has_layout_image() -> None:
    """plot_clusters adds the source image as a layout image."""
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    labels = np.zeros((8, 10), dtype=np.int32)
    fig = plot_clusters(image, labels, bit_depth=8)
    assert len(fig.layout.images) > 0


@pytest.mark.parametrize("n_clusters", [2, 5])
def test_plot_clusters_title_contains_k(n_clusters: int) -> None:
    """Figure title contains the number of clusters."""
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    labels = (np.arange(6 * 8) % n_clusters).reshape(6, 8).astype(np.int32)
    fig = plot_clusters(image, labels, bit_depth=8)
    assert str(n_clusters) in str(fig.layout.title.text)


# ---------------------------------------------------------------------------
# plot_mask (module-level function)
# ---------------------------------------------------------------------------


def test_plot_mask_returns_figure() -> None:
    """plot_mask returns a Plotly Figure."""
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    mask = np.zeros((12, 16), dtype=np.bool_)
    mask[:6, :] = True
    fig = plot_mask(image, mask, bit_depth=8)
    assert isinstance(fig, go.Figure)


def test_plot_mask_has_overlay_trace() -> None:
    """plot_mask includes a go.Image trace for the mask overlay."""
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    mask = np.zeros((8, 10), dtype=np.bool_)
    fig = plot_mask(image, mask, bit_depth=8)
    assert any(isinstance(t, go.Image) for t in fig.data)


def test_plot_mask_has_layout_image() -> None:
    """plot_mask adds the source image as a layout image."""
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    mask = np.zeros((8, 10), dtype=np.bool_)
    fig = plot_mask(image, mask, bit_depth=8)
    assert len(fig.layout.images) > 0


# ---------------------------------------------------------------------------
# MaskBuilder.plot_clusters / MaskBuilder.plot_mask
# ---------------------------------------------------------------------------


def test_mb_plot_clusters_returns_figure(builder: MaskBuilder) -> None:
    """MaskBuilder.plot_clusters returns a Figure after clustering."""
    assert isinstance(builder.plot_clusters(), go.Figure)


def test_mb_plot_clusters_before_clustering_raises(rgb_frame: Frame) -> None:
    """MaskBuilder.plot_clusters raises MaskError before clustering."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(MaskError, match="compute_clusters"):
        mb.plot_clusters()


def test_mb_plot_mask_returns_figure(builder: MaskBuilder) -> None:
    """MaskBuilder.plot_mask returns a Figure after mask creation."""
    builder.apply_labels({0})
    assert isinstance(builder.plot_mask(), go.Figure)


def test_mb_plot_mask_before_mask_raises(rgb_frame: Frame) -> None:
    """MaskBuilder.plot_mask raises MaskError before mask creation."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(MaskError, match="mask must exist"):
        mb.plot_mask()
