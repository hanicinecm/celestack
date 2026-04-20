"""Tests for mask plotting helpers."""

import numpy as np
import plotly.graph_objects as go
import pytest

from celestack.mask._plotting import plot_mask


def test_plot_mask_returns_figure() -> None:
    """plot_mask returns a Plotly Figure."""
    mask = np.zeros((12, 16), dtype=np.bool_)
    mask[:6, :] = True
    fig = plot_mask(mask, title="<Mask>", highlight_noise=0)
    assert isinstance(fig, go.Figure)


def test_plot_mask_has_layout_image() -> None:
    """plot_mask adds the mask image as a layout image."""
    mask = np.zeros((8, 10), dtype=np.bool_)
    fig = plot_mask(mask, title="<Mask>", highlight_noise=0)
    assert len(fig.layout.images) > 0


def test_plot_mask_has_no_traces() -> None:
    """plot_mask adds no traces at all."""
    mask = np.zeros((8, 10), dtype=np.bool_)
    fig = plot_mask(mask, title="<Mask>", highlight_noise=0)
    assert len(fig.data) == 0


def test_plot_mask_has_no_image_trace() -> None:
    """plot_mask does not use go.Image traces."""
    mask = np.zeros((8, 10), dtype=np.bool_)
    fig = plot_mask(mask, title="<Mask>", highlight_noise=0)
    assert not any(isinstance(t, go.Image) for t in fig.data)


def test_plot_mask_custom_title() -> None:
    """plot_mask uses the provided title."""
    mask = np.zeros((8, 10), dtype=np.bool_)
    fig = plot_mask(mask, title="My Mask", highlight_noise=0)
    assert "My Mask" in str(fig.layout.title.text)


def test_plot_mask_axes_match_array_shape() -> None:
    """plot_mask sets axis ranges to the mask's native pixel dimensions."""
    mask = np.zeros((6, 8), dtype=np.bool_)
    fig = plot_mask(mask, title="<Mask>", highlight_noise=0)
    assert fig.layout.xaxis.range[1] == 8
    assert fig.layout.yaxis.range[0] == 6


def test_plot_mask_highlight_noise_adds_fg_trace() -> None:
    """highlight_noise adds a Scatter trace for foreground noise."""
    mask = np.zeros((20, 20), dtype=np.bool_)
    mask[5, 5] = True
    fig = plot_mask(mask, title="<Mask>", highlight_noise=1)
    names = {t.name for t in fig.data}
    assert "FG noise" in names


def test_plot_mask_highlight_noise_adds_bg_trace() -> None:
    """highlight_noise adds a Scatter trace for background noise."""
    mask = np.ones((20, 20), dtype=np.bool_)
    mask[5, 5] = False
    fig = plot_mask(mask, title="<Mask>", highlight_noise=1)
    names = {t.name for t in fig.data}
    assert "BG noise" in names


def test_plot_mask_highlight_noise_zero_no_traces() -> None:
    """highlight_noise=0 disables Scatter traces."""
    mask = np.zeros((20, 20), dtype=np.bool_)
    mask[5, 5] = True
    fig = plot_mask(mask, title="<Mask>", highlight_noise=0)
    assert len(fig.data) == 0


def test_plot_mask_highlight_noise_no_particles_no_traces() -> None:
    """When no small particles exist, no traces are added."""
    mask = np.ones((20, 20), dtype=np.bool_)
    fig = plot_mask(mask, title="<Mask>", highlight_noise=1)
    assert len(fig.data) == 0


def test_plot_mask_highlight_noise_marker_sizes_proportional() -> None:
    """Larger particles get larger markers."""
    mask = np.zeros((30, 30), dtype=np.bool_)
    mask[5, 5] = True
    mask[15, 15:18] = True
    fig = plot_mask(mask, title="<Mask>", highlight_noise=5)
    trace = next(t for t in fig.data if t.name == "FG noise")
    sizes = trace.marker.size
    assert max(sizes) > min(sizes)


def test_plot_mask_highlight_noise_only_one_layout_image() -> None:
    """highlight_noise does not add a second layout image."""
    mask = np.zeros((20, 20), dtype=np.bool_)
    mask[5, 5] = True
    fig = plot_mask(mask, title="<Mask>", highlight_noise=1)
    assert len(fig.layout.images) == 1


def test_plot_mask_highlight_noise_coordinates_in_native_pixels() -> None:
    """Scatter coordinates are in the mask's native pixel space."""
    mask = np.zeros((20, 20), dtype=np.bool_)
    mask[10, 10] = True
    fig = plot_mask(mask, title="<Mask>", highlight_noise=1)
    trace = next(t for t in fig.data if t.name == "FG noise")
    assert trace.x[0] == pytest.approx(10.0)
    assert trace.y[0] == pytest.approx(10.0)
