"""Cross-cutting progress reporting.

Provides a minimal progress-bar protocol and a module-level factory
that can be swapped once at startup (e.g. by a GUI layer).  Domain
code calls ``progress_factory()`` — callers never thread a factory
through every function signature.
"""

from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
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
    """Minimal progress-bar interface.

    Usable as a context manager: ``__exit__`` calls :meth:`close`,
    guaranteeing teardown even if the enclosed block raises.
    """

    def update(self, n: int = 1) -> None:
        """Advance the bar by *n* steps."""
        ...

    def close(self) -> None:
        """Finalize the bar."""
        ...

    def __enter__(self) -> ProgressBar: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


type ProgressFactory = Callable[[int | None, str], ProgressBar]


class _RichBar:
    """Thin wrapper around a *rich* Progress display.

    When *total* is ``None``, the bar runs in indeterminate mode (pulsing
    bar + animated spinner) without completion count or ETA columns.
    """

    def __init__(self, total: int | None, description: str) -> None:
        columns: tuple = (
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
        )
        if total is not None:
            columns = (
                *columns,
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
        else:
            columns = (*columns, TimeElapsedColumn())
        self._progress = Progress(*columns)
        self._progress.start()
        self._task = self._progress.add_task(description, total=total)

    def update(self, n: int = 1) -> None:
        self._progress.update(self._task, advance=n)

    def close(self) -> None:
        if self._progress.tasks[self._task].total is None:
            self._progress.update(self._task, total=1, completed=1)  # clears spinner

        self._progress.stop()

    def __enter__(self) -> _RichBar:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _rich_factory(total: int | None, description: str) -> ProgressBar:
    """Default factory: terminal progress bar via *rich*."""
    return _RichBar(total, description)


class _NoopBar:
    """Silent progress bar that also serves as its own factory."""

    def __call__(
        self,
        total: int | None,  # noqa: ARG002
        description: str,  # noqa: ARG002
    ) -> _NoopBar:
        return self

    def update(self, n: int = 1) -> None:  # noqa: ARG002
        pass

    def close(self) -> None:
        pass

    def __enter__(self) -> _NoopBar:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pass


_FACTORY: ProgressFactory = _rich_factory


def set_progress_factory(factory: ProgressFactory | None) -> None:
    """Replace the global progress-bar factory.

    Pass ``None`` to silence all progress output.
    """
    global _FACTORY
    _FACTORY = factory if factory is not None else _NoopBar()


def progress_factory(total: int | None, description: str = "") -> ProgressBar:
    """Create a progress bar using the current global factory."""
    return _FACTORY(total, description)
