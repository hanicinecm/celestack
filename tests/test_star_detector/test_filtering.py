"""Tests for post-detection filtering."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from celestack.star_detector._filtering import filter_stars


def _star_df(rows: list[tuple[float, float, float]]) -> pl.DataFrame:
    """Build a minimal star DataFrame from (x, y, flux_local) tuples."""
    return pl.DataFrame(
        {
            "x": [r[0] for r in rows],
            "y": [r[1] for r in rows],
            "flux_local": [r[2] for r in rows],
        }
    )


@pytest.fixture()
def simple_mask() -> np.ndarray:
    """64x64 mask: bottom 16 rows are foreground."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[48:, :] = True
    return mask


def test_proximity_filter_removes_faint_duplicate(simple_mask: np.ndarray):
    """Two stars within min_separation: fainter one is removed."""
    stars = _star_df([(30.0, 20.0, 100.0), (32.0, 20.0, 50.0)])
    result = filter_stars(
        stars,
        simple_mask,
        target_stars=10,
        min_separation=5.0,
        edge_margin=1,
    )
    assert len(result) == 1
    assert result["flux_local"][0] == 100.0


def test_edge_margin_removes_near_frame_edge(simple_mask: np.ndarray):
    """Star within edge_margin of frame edge is removed."""
    stars = _star_df([(2.0, 20.0, 100.0), (30.0, 20.0, 80.0)])
    result = filter_stars(
        stars,
        simple_mask,
        target_stars=10,
        min_separation=1.0,
        edge_margin=5,
    )
    assert len(result) == 1
    assert result["flux_local"][0] == 80.0


def test_mask_boundary_exclusion(simple_mask: np.ndarray):
    """Star near the foreground mask boundary is excluded."""
    # Row 47 in proxy is right at boundary with foreground starting at 48
    stars = _star_df([(30.0, 47.0, 100.0), (30.0, 20.0, 80.0)])
    result = filter_stars(
        stars,
        simple_mask,
        target_stars=10,
        min_separation=1.0,
        edge_margin=5,
    )
    assert len(result) == 1
    assert result["flux_local"][0] == 80.0


def test_cap_keeps_brightest(simple_mask: np.ndarray):
    """Cap keeps only the target_stars brightest stars."""
    rows = [(10.0 + i * 5, 20.0, float(100 - i)) for i in range(10)]
    stars = _star_df(rows)
    result = filter_stars(
        stars,
        simple_mask,
        target_stars=5,
        min_separation=1.0,
        edge_margin=1,
    )
    assert len(result) == 5
    fluxes = result["flux_local"].to_list()
    assert fluxes == sorted(fluxes, reverse=True)


def test_empty_input_returns_empty(simple_mask: np.ndarray):
    """Empty input DataFrame returns empty output."""
    stars = pl.DataFrame({"x": [], "y": [], "flux_local": []})
    result = filter_stars(
        stars,
        simple_mask,
        target_stars=10,
        min_separation=5.0,
        edge_margin=5,
    )
    assert len(result) == 0


def test_star_within_proxy_frame_bounds_is_kept():
    """Star within proxy frame bounds passes the edge exclusion check."""
    mask = np.zeros((32, 32), dtype=bool)
    # Star at x=20 is well within the 32-px proxy width.
    stars = _star_df([(20.0, 10.0, 100.0)])
    result = filter_stars(
        stars,
        mask,
        target_stars=10,
        min_separation=1.0,
        edge_margin=2,
    )
    assert len(result) == 1
