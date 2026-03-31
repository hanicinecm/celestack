"""Tests for frame plotting helpers."""

import numpy as np
import pytest

from celestack.frame._plotting import as_png_data_uri


def test_as_png_data_uri_invalid_shape() -> None:
    """1D array raises ValueError."""
    arr = np.zeros(10, dtype=np.uint8)
    with pytest.raises(ValueError, match="Unsupported array shape"):
        as_png_data_uri(arr)
