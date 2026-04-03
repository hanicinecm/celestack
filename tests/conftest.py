"""Root test configuration."""

from collections.abc import Iterator

import pytest

import celestack.progress as _progress_mod
from celestack.progress import set_progress_factory


@pytest.fixture(autouse=True)
def _silence_progress() -> Iterator[None]:
    """Disable progress bars during tests."""
    original = _progress_mod._FACTORY
    set_progress_factory(None)
    yield
    _progress_mod._FACTORY = original
