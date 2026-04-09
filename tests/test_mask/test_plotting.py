"""Tests for mask plotting helpers."""

import numpy as np
import plotly.graph_objects as go
import pytest

from celestack.exceptions import MaskError
from celestack.frame.core import Frame
from celestack.mask._mask_builder import MaskBuilder
from celestack.mask._plotting import plot_clusters, plot_mask

# ---------------------------------------------------------------------------
# plot_clusters (module-level function)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# plot_mask (module-level function)
# ---------------------------------------------------------------------------


def test_plot_mask_returns_figure() -> None:
    """plot_mask returns a Plotly Figure."""
    mask = np.zeros((12, 16), dtype=np.bool_)
    mask[:6, :] = True
    fig = plot_mask(mask, title="<Mask>", downscale_factor=1, highlight_noise=0)
    assert isinstance(fig, go.Figure)


def test_plot_mask_has_layout_image() -> None:
    """plot_mask adds the mask image as a layout image."""
    mask = np.zeros((8, 10), dtype=np.bool_)
    fig = plot_mask(mask, title="<Mask>", downscale_factor=1, highlight_noise=0)
    assert len(fig.layout.images) > 0


def test_plot_mask_has_no_traces() -> None:
    """plot_mask adds no traces at all."""
    mask = np.zeros((8, 10), dtype=np.bool_)
    fig = plot_mask(mask, title="<Mask>", downscale_factor=1, highlight_noise=0)
    assert len(fig.data) == 0


def test_plot_mask_has_no_image_trace() -> None:
    """plot_mask does not use go.Image traces."""
    mask = np.zeros((8, 10), dtype=np.bool_)
    fig = plot_mask(mask, title="<Mask>", downscale_factor=1, highlight_noise=0)
    assert not any(isinstance(t, go.Image) for t in fig.data)


def test_plot_mask_custom_title() -> None:
    """plot_mask uses the provided title."""
    mask = np.zeros((8, 10), dtype=np.bool_)
    fig = plot_mask(mask, title="My Mask", downscale_factor=1, highlight_noise=0)
    assert "My Mask" in str(fig.layout.title.text)


def test_plot_mask_downscale_factor_scales_axes() -> None:
    """plot_mask scales axes to full-resolution dimensions."""
    mask = np.zeros((6, 8), dtype=np.bool_)
    fig = plot_mask(mask, title="<Mask>", downscale_factor=4, highlight_noise=0)
    assert fig.layout.xaxis.range[1] == 32
    assert fig.layout.yaxis.range[0] == 24


# ---------------------------------------------------------------------------
# plot_mask — highlight_noise
# ---------------------------------------------------------------------------


def test_plot_mask_highlight_noise_adds_fg_trace() -> None:
    """highlight_noise adds a Scatter trace for foreground noise."""
    mask = np.zeros((20, 20), dtype=np.bool_)
    mask[5, 5] = True  # single-pixel foreground particle
    fig = plot_mask(mask, title="<Mask>", downscale_factor=1, highlight_noise=1)
    names = {t.name for t in fig.data}
    assert "FG noise" in names


def test_plot_mask_highlight_noise_adds_bg_trace() -> None:
    """highlight_noise adds a Scatter trace for background noise."""
    mask = np.ones((20, 20), dtype=np.bool_)
    mask[5, 5] = False  # single-pixel background hole
    fig = plot_mask(mask, title="<Mask>", downscale_factor=1, highlight_noise=1)
    names = {t.name for t in fig.data}
    assert "BG noise" in names


def test_plot_mask_highlight_noise_zero_no_traces() -> None:
    """highlight_noise=0 disables Scatter traces."""
    mask = np.zeros((20, 20), dtype=np.bool_)
    mask[5, 5] = True
    fig = plot_mask(mask, title="<Mask>", downscale_factor=1, highlight_noise=0)
    assert len(fig.data) == 0


def test_plot_mask_highlight_noise_no_particles_no_traces() -> None:
    """When no small particles exist, no traces are added."""
    mask = np.ones((20, 20), dtype=np.bool_)
    fig = plot_mask(mask, title="<Mask>", downscale_factor=1, highlight_noise=1)
    assert len(fig.data) == 0


def test_plot_mask_highlight_noise_marker_sizes_proportional() -> None:
    """Larger particles get larger markers."""
    mask = np.zeros((30, 30), dtype=np.bool_)
    mask[5, 5] = True  # 1-pixel particle
    mask[15, 15:18] = True  # 3-pixel particle
    fig = plot_mask(mask, title="<Mask>", downscale_factor=1, highlight_noise=5)
    trace = next(t for t in fig.data if t.name == "FG noise")
    sizes = trace.marker.size
    # The 3-pixel particle should have a larger marker than the 1-pixel one
    assert max(sizes) > min(sizes)


def test_plot_mask_highlight_noise_only_one_layout_image() -> None:
    """highlight_noise does not add a second layout image."""
    mask = np.zeros((20, 20), dtype=np.bool_)
    mask[5, 5] = True
    fig = plot_mask(mask, title="<Mask>", downscale_factor=1, highlight_noise=1)
    assert len(fig.layout.images) == 1


def test_plot_mask_highlight_noise_scales_coordinates() -> None:
    """Scatter coordinates account for downscale_factor."""
    mask = np.zeros((20, 20), dtype=np.bool_)
    mask[10, 10] = True  # centroid at (10, 10) in proxy space
    fig = plot_mask(mask, title="<Mask>", downscale_factor=4, highlight_noise=1)
    trace = next(t for t in fig.data if t.name == "FG noise")
    assert trace.x[0] == pytest.approx(40.0)
    assert trace.y[0] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# MaskBuilder.plot_clusters
# ---------------------------------------------------------------------------


def test_mb_plot_clusters_returns_figure(builder: MaskBuilder) -> None:
    """MaskBuilder.plot_clusters returns a Figure after clustering."""
    assert isinstance(builder.plot_clusters(), go.Figure)


def test_mb_plot_clusters_before_clustering_raises(rgb_frame: Frame) -> None:
    """MaskBuilder.plot_clusters raises MaskError before clustering."""
    mb = MaskBuilder(rgb_frame)
    with pytest.raises(MaskError, match="compute_clusters"):
        mb.plot_clusters()
