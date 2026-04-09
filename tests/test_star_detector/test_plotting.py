"""Smoke tests for star detection plotting."""

from __future__ import annotations

import plotly.graph_objects as go

from celestack.frame.core import Frame
from celestack.mask.core import Mask
from celestack.star_detector.core import StarDetector


def test_plot_returns_figure(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """plot() returns a Plotly Figure after detect()."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    sd.detect(target_stars=5, n_segments=2)
    fig = sd.plot()
    assert isinstance(fig, go.Figure)


def test_plot_with_segments_returns_figure(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """plot(show_segments=True) returns a Plotly Figure."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    sd.detect(target_stars=5, n_segments=2)
    fig = sd.plot(show_segments=True)
    assert isinstance(fig, go.Figure)
