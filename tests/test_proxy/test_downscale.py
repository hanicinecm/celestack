"""Tests for block_average and majority_vote."""

import numpy as np
import pytest

from celestack.proxy._downscale import block_average, majority_vote


def test_block_average_2d() -> None:
    """2D block averaging produces the mean of each 2x2 block."""
    arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    result = block_average(arr, 2)
    np.testing.assert_allclose(result, [[3.5, 5.5]])


def test_block_average_3d() -> None:
    """3D arrays downscale per channel."""
    arr = np.ones((4, 4, 3), dtype=np.float32)
    arr[..., 0] = 2.0
    result = block_average(arr, 2)
    assert result.shape == (2, 2, 3)
    np.testing.assert_allclose(result[..., 0], 2.0)
    np.testing.assert_allclose(result[..., 1], 1.0)


def test_block_average_trims_remainder() -> None:
    """Trailing partial rows/columns are trimmed."""
    arr = np.ones((5, 5), dtype=np.float32)
    result = block_average(arr, 2)
    assert result.shape == (2, 2)


def test_block_average_factor_one_copies() -> None:
    """factor=1 returns a copy, not the same array."""
    arr = np.arange(4, dtype=np.float32).reshape(2, 2)
    result = block_average(arr, 1)
    np.testing.assert_array_equal(result, arr)
    assert result is not arr


def test_block_average_invalid_factor() -> None:
    """factor < 1 raises ValueError."""
    with pytest.raises(ValueError, match=">= 1"):
        block_average(np.zeros((4, 4)), 0)


def test_majority_vote_strict_threshold() -> None:
    """Blocks with exactly 50% foreground become False (strict > 0.5)."""
    mask = np.array(
        [
            [True, True, False, False],
            [False, False, False, False],
        ]
    )
    result = majority_vote(mask, 2)
    assert result.shape == (1, 2)
    assert not result.any()


def test_majority_vote_majority_true() -> None:
    """Blocks with > 50% True become True."""
    mask = np.array(
        [
            [True, True, False, True],
            [True, False, False, False],
        ]
    )
    result = majority_vote(mask, 2)
    assert result[0, 0]
    assert not result[0, 1]


def test_majority_vote_factor_one_copies() -> None:
    """factor=1 returns a copy."""
    mask = np.array([[True, False]])
    result = majority_vote(mask, 1)
    np.testing.assert_array_equal(result, mask)
    assert result is not mask


def test_majority_vote_rejects_non_bool() -> None:
    """Non-boolean input raises ValueError."""
    with pytest.raises(ValueError, match="2D boolean"):
        majority_vote(np.zeros((4, 4), dtype=np.uint8), 2)


def test_majority_vote_rejects_3d() -> None:
    """3D input raises ValueError."""
    with pytest.raises(ValueError, match="2D boolean"):
        majority_vote(np.zeros((4, 4, 3), dtype=np.bool_), 2)


def test_majority_vote_invalid_factor() -> None:
    """factor < 1 raises ValueError."""
    with pytest.raises(ValueError, match=">= 1"):
        majority_vote(np.zeros((4, 4), dtype=np.bool_), 0)
