"""Cross-cutting progress reporting.

Provides a minimal progress-bar protocol and a module-level factory
that can be swapped once at startup (e.g. by a GUI layer).  Domain
code calls ``progress_factory()`` — callers never thread a factory
through every function signature.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressBar(Protocol):
    """Minimal progress-bar interface."""

    def update(self, n: int = 1) -> None:
        """Advance the bar by *n* steps."""
        ...

    def close(self) -> None:
        """Finalize the bar."""
        ...


type ProgressFactory = Callable[[int, str], ProgressBar]


class _RichBar:
    """Thin wrapper around a *rich* Progress display."""

    def __init__(self, total: int, description: str) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
        self._progress.start()
        self._task = self._progress.add_task(description, total=total)

    def update(self, n: int = 1) -> None:
        self._progress.update(self._task, advance=n)

    def close(self) -> None:
        self._progress.stop()


def _rich_factory(total: int, description: str) -> ProgressBar:
    """Default factory: terminal progress bar via *rich*."""
    return _RichBar(total, description)


class _NoopBar:
    """Silent progress bar that also serves as its own factory."""

    def __call__(self, total: int, description: str) -> _NoopBar:  # noqa: ARG002
        return self

    def update(self, n: int = 1) -> None:  # noqa: ARG002
        pass

    def close(self) -> None:
        pass


_FACTORY: ProgressFactory = _rich_factory


def set_progress_factory(factory: ProgressFactory | None) -> None:
    """Replace the global progress-bar factory.

    Pass ``None`` to silence all progress output.
    """
    global _FACTORY
    _FACTORY = factory if factory is not None else _NoopBar()


def progress_factory(total: int, description: str = "") -> ProgressBar:
    """Create a progress bar using the current global factory."""
    return _FACTORY(total, description)
