"""Tests for mask morphological operations."""

from __future__ import annotations

import numpy as np
import pytest

from celestack.mask._morphology import remove_small_components


def test_single_pixel_foreground_removed():
    """A single foreground pixel in an otherwise empty mask is removed."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[5, 5] = True
    result = remove_small_components(mask, max_size=1)
    assert not result.any()


def test_single_pixel_background_filled():
    """A single background pixel in an otherwise full mask is filled."""
    mask = np.ones((10, 10), dtype=bool)
    mask[5, 5] = False
    result = remove_small_components(mask, max_size=1)
    assert result.all()


def test_large_foreground_preserved():
    """A foreground blob larger than max_size survives."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True  # 9 pixels
    result = remove_small_components(mask, max_size=5)
    assert result[2:5, 2:5].all()


def test_large_background_preserved():
    """A background hole larger than max_size survives."""
    mask = np.ones((10, 10), dtype=bool)
    mask[2:5, 2:5] = False  # 9 pixels
    result = remove_small_components(mask, max_size=5)
    assert not result[2:5, 2:5].any()


def test_component_at_max_size_removed():
    """A component with area exactly equal to max_size is removed."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0:3] = True  # 3 pixels in a row (connected)
    result = remove_small_components(mask, max_size=3)
    assert not result.any()


def test_component_above_max_size_preserved():
    """A component with area max_size + 1 survives."""
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0:4] = True  # 4 pixels
    result = remove_small_components(mask, max_size=3)
    assert result[0, 0:4].all()


def test_both_phases_cleaned():
    """Small foreground and background particles are both removed."""
    mask = np.zeros((10, 10), dtype=bool)
    # Large foreground block
    mask[0:5, 0:10] = True
    # Small foreground particle in background region
    mask[7, 7] = True
    # Small background hole in foreground region
    mask[2, 2] = False

    result = remove_small_components(mask, max_size=1)

    # Large block preserved
    assert result[0:5, 0:10].sum() == 50
    # Small foreground particle removed
    assert not result[7, 7]
    # Small background hole filled
    assert result[2, 2]


def test_all_false_unchanged():
    """An all-background mask is unchanged."""
    mask = np.zeros((10, 10), dtype=bool)
    result = remove_small_components(mask, max_size=5)
    assert not result.any()


def test_all_true_unchanged():
    """An all-foreground mask is unchanged."""
    mask = np.ones((10, 10), dtype=bool)
    result = remove_small_components(mask, max_size=5)
    assert result.all()


def test_returns_copy():
    """The result is a new array, not the input."""
    mask = np.zeros((10, 10), dtype=bool)
    result = remove_small_components(mask, max_size=1)
    assert result is not mask


def test_four_connectivity():
    """Diagonally adjacent pixels are separate components under 4-connectivity."""
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True  # diagonal neighbor — not connected
    result = remove_small_components(mask, max_size=1)
    # Both are single-pixel components and should be removed
    assert not result.any()


@pytest.mark.parametrize(
    ("shape", "max_size"),
    [
        ((1, 1), 1),
        ((100, 100), 10),
        ((5, 200), 3),
    ],
)
def test_various_shapes(shape: tuple[int, int], max_size: int):
    """The function handles various mask dimensions without error."""
    mask = np.random.default_rng(42).random(shape) > 0.5
    result = remove_small_components(mask, max_size)
    assert result.shape == shape
    assert result.dtype == np.bool_
