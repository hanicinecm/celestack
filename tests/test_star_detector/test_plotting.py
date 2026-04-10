"""Smoke tests for star detection plotting."""

from __future__ import annotations

import plotly.graph_objects as go

from celestack.frame.core import Frame
from celestack.mask.core import Mask
from celestack.star_detector.core import StarDetector


def test_plot_returns_figure_before_any_step(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """plot() returns a Figure even before segment() or detect()."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    assert isinstance(sd.plot(), go.Figure)
    assert isinstance(sd.plot(show_segments=True), go.Figure)


def test_plot_returns_figure_after_segment(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """plot(show_segments=True) returns a Figure after segment() only."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    sd.segment(n_segments=2)
    assert isinstance(sd.plot(show_segments=True), go.Figure)


def test_plot_returns_figure_after_detect(
    full_res_frame: Frame,
    sky_mask_top_half: Mask,
):
    """plot() returns a Figure with stars after detect()."""
    sd = StarDetector(full_res_frame, sky_mask_top_half)
    sd.segment(n_segments=2)
    sd.detect(target_stars=5)
    assert isinstance(sd.plot(), go.Figure)
    assert isinstance(sd.plot(show_segments=True), go.Figure)
