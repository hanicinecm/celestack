"""Tests for proxy frame plotting."""

import numpy as np
import plotly.graph_objects as go

from celestack.frame.core import Frame
from celestack.mask.core import Mask
from celestack.proxy._plotting import plot_proxy_frame
from celestack.proxy.frame import ProxyFrame


def test_plot_proxy_frame_returns_figure() -> None:
    """plot_proxy_frame returns a Plotly Figure with a layout image."""
    array = np.zeros((16, 16), dtype=np.float16)
    array[4:12, 4:12] = 0.2
    fig = plot_proxy_frame(array, title="t")
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.images) == 1


def test_plot_proxy_frame_all_zero() -> None:
    """All-zero arrays still render without error."""
    array = np.zeros((8, 8), dtype=np.float16)
    fig = plot_proxy_frame(array, title="zero")
    assert isinstance(fig, go.Figure)


def test_proxyframe_plot(rgb_frame: Frame, sky_mask: Mask) -> None:
    """ProxyFrame.plot returns a Plotly figure."""
    pf = ProxyFrame.from_masked_frame(
        rgb_frame, sky_mask, downscale_factor=4, box_size=8, filter_size=1
    )
    fig = pf.plot()
    assert isinstance(fig, go.Figure)
