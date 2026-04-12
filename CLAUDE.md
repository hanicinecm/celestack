# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Celestack is a Python library for stacking untracked starscape astrophotography images. See [SPEC.md](SPEC.md) for the full high-level specification, including architecture, workflow steps, dependency graph, and design decisions.

## General Principles

- When in doubt, ask before making assumptions.
- Strive for elegant, Pythonic, and modular solutions.
- Never discard user changes. If conflicts arise, ask before rewriting.
- Do not commit changes unless asked.
- Use the [Conventional Commits](https://www.conventionalcommits.org/) standard for all commit messages (e.g., `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`).
- Include a brief body in commit messages summarizing what was changed, with slightly more detail than the title line.
- Always push local commits to the remote after committing.

## Project Layout

- `src/celestack/` — source package (standard `src` layout)
- `tests/` — mirrors source layout (e.g., `src/celestack/backend/foo.py` → `tests/test_backend/test_foo.py`)
- `.venv/` — virtual environment managed by `uv`

## Commands

All commands that rely on the virtual environment bening active must be prefixed by `uv run`.
Examples:

- **Format and lint**: `uv run ruff check --fix && uv run ruff format` (always run on the entire codebase)
- **Run all tests**: `uv run pytest`
- **Run a single test**: `uv run pytest tests/test_module.py::test_name`
- **Install a dependency**: `uv add <pkg>` (or `uv add --dev <pkg>` for dev dependencies)

The project uses `uv` for environment management but must remain installable with `pip`.

## Markdown Style

- Always add spaces around `|` in tables, including separator rows: `| --- | --- |` not `|---|---|`.
- Surround lists with blank lines (before and after).
- Surround fenced code blocks with blank lines (before and after).

## Coding Style

- 4-space indentation, 88-character line limit (ruff/Black default).
- Type hints everywhere — public and private interfaces. Use native types (`tuple[str]`), not `typing` aliases (`Tuple[str]`).
- Google-style docstrings without types (rely on type hints). Document exceptions. Add module-level docstrings.
- Favor short, composable functions over deep inheritance.
- Prefer `pathlib` over `os` for path manipulation.
- Use absolute imports only (no relative imports).
- Define `__all__` only where namespace control is required (e.g., package `__init__` files).
- Pre-define exception messages in a `msg` variable before raising:

```python
# Do this:
msg = "Error message"
raise ValueError(msg)

# Not this:
raise ValueError("Error message")
```

### Docstring Format

```python
def load_manifest(path: Path) -> Manifest:
    """Load a manifest file into memory.

    Optional more detailed overview.

    Args:
        path: Location of the manifest file.

    Returns:
        Parsed manifest instance.

    Raises:
        ManifestError: If the file is unreadable or malformed.
    """
```

## Testing

- Prefer test functions over test classes.
- One-liner docstring per test function (no full Google-style needed).
- Use `pytest` fixtures; organize shared fixtures in `conftest.py`.
- Parametrize tests to improve coverage.
- Assert exceptions with the `match` parameter of `pytest.raises`:

```python
def test_invalid_input():
    """Reject malformed input with a clear error."""
    with pytest.raises(ValueError, match=r"no such file exists"):
        check_file_exists()
```

- Maintain high test coverage on modified code; add tests for new or changed behavior unless instructed otherwise.
