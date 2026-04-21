"""Tests for proxy frame and proxy mask plotting."""

import numpy as np
import plotly.graph_objects as go

from celestack.frame.core import Frame
from celestack.mask.core import Mask
from celestack.proxy._plotting import plot_proxy_frame, plot_proxy_mask
from celestack.proxy.frame import ProxyFrame
from celestack.proxy.mask import ProxyMask


def test_plot_proxy_frame_show_pixels_heatmap() -> None:
    """Default show_pixels=True renders a Heatmap trace, no layout image."""
    array = np.zeros((16, 16), dtype=np.float16)
    array[4:12, 4:12] = 0.2
    fig = plot_proxy_frame(array, title="t", downscale_factor=4)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Heatmap)
    assert len(fig.layout.images) == 0


def test_plot_proxy_frame_layout_image_mode() -> None:
    """show_pixels=False renders a static PNG layout image, no traces."""
    array = np.zeros((16, 16), dtype=np.float16)
    fig = plot_proxy_frame(array, title="t", downscale_factor=4, show_pixels=False)
    assert len(fig.data) == 0
    assert len(fig.layout.images) == 1


def test_plot_proxy_frame_upscale_coordinates() -> None:
    """upscale_coordinates=True scales axis range by the downscale factor."""
    array = np.zeros((8, 8), dtype=np.float16)
    fig = plot_proxy_frame(array, title="t", downscale_factor=4)
    assert tuple(fig.layout.xaxis.range) == (0, 32)
    assert tuple(fig.layout.yaxis.range) == (32, 0)
    assert fig.data[0].dx == 4


def test_plot_proxy_frame_native_coordinates() -> None:
    """upscale_coordinates=False keeps axes in proxy pixel space."""
    array = np.zeros((8, 8), dtype=np.float16)
    fig = plot_proxy_frame(
        array, title="t", downscale_factor=4, upscale_coordinates=False
    )
    assert tuple(fig.layout.xaxis.range) == (0, 8)
    assert fig.data[0].dx == 1


def test_plot_proxy_frame_all_zero() -> None:
    """All-zero arrays still render without error."""
    array = np.zeros((8, 8), dtype=np.float16)
    fig = plot_proxy_frame(array, title="zero", downscale_factor=2)
    assert isinstance(fig, go.Figure)


def test_proxyframe_plot_defaults(rgb_frame: Frame, sky_mask: Mask) -> None:
    """ProxyFrame.plot defaults to heatmap + full-res coordinates."""
    pf = ProxyFrame.from_masked_frame(
        rgb_frame, sky_mask, downscale_factor=4, box_size=8, filter_size=1
    )
    fig = pf.plot()
    assert isinstance(fig, go.Figure)
    assert isinstance(fig.data[0], go.Heatmap)
    assert fig.data[0].dx == 1


def test_proxyframe_plot_image_mode(rgb_frame: Frame, sky_mask: Mask) -> None:
    """ProxyFrame.plot with show_pixels=False falls back to layout image."""
    pf = ProxyFrame.from_masked_frame(
        rgb_frame, sky_mask, downscale_factor=4, box_size=8, filter_size=1
    )
    fig = pf.plot(show_pixels=False)
    assert len(fig.data) == 0
    assert len(fig.layout.images) == 1


def test_plot_proxy_mask_show_pixels_heatmap() -> None:
    """plot_proxy_mask default renders a Heatmap of the boolean values."""
    array = np.zeros((8, 8), dtype=np.bool_)
    array[:4, :] = True
    fig = plot_proxy_mask(array, title="m", downscale_factor=3)
    assert isinstance(fig.data[0], go.Heatmap)
    assert fig.data[0].dx == 3


def test_plot_proxy_mask_layout_image_mode() -> None:
    """plot_proxy_mask with show_pixels=False renders a layout image."""
    array = np.zeros((8, 8), dtype=np.bool_)
    fig = plot_proxy_mask(array, title="m", downscale_factor=2, show_pixels=False)
    assert len(fig.data) == 0
    assert len(fig.layout.images) == 1


def test_proxymask_plot_defaults() -> None:
    """ProxyMask.plot returns a heatmap with full-res coordinates by default."""
    mask = Mask._from_array(np.ones((16, 16), dtype=np.bool_))
    pm = ProxyMask.from_mask(mask, 4)
    fig = pm.plot()
    assert isinstance(fig.data[0], go.Heatmap)
    assert tuple(fig.layout.xaxis.range) == (0, 16)
