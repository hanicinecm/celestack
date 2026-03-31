"""Averaging method implementations for frame stacking."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Method(Protocol):
    """Protocol for pixel-combining methods."""

    NAME: str

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        """Combine frames stacked along axis 0 into a single result.

        Args:
            stack: Array with shape ``(N, H, W, ...)`` where *N* is the
                number of frames.

        Returns:
            Combined array with shape ``(H, W, ...)``.
        """
        ...


class Mean:
    """Combine along axis 0 using the arithmetic mean."""

    NAME: str = "mean"

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        """Return the per-pixel mean, preserving the input dtype."""
        return np.mean(stack, axis=0).astype(stack.dtype)


class Median:
    """Combine along axis 0 using the median."""

    NAME: str = "median"

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        """Return the per-pixel median, preserving the input dtype."""
        return np.median(stack, axis=0).astype(stack.dtype)


class SigmaClip:
    """Combine along axis 0 using sigma-clipped mean.

    Args:
        kappa: Number of standard deviations for clipping.
    """

    NAME: str = "sigma_clip"

    def __init__(self, kappa: float = 3.0) -> None:
        self._kappa = kappa

    def __call__(self, stack: np.ndarray) -> np.ndarray:
        """Return the sigma-clipped mean, preserving the input dtype."""
        mean = np.mean(stack, axis=0, keepdims=True)
        std = np.std(stack, axis=0, keepdims=True)
        mask = np.abs(stack - mean) <= self._kappa * std
        clipped_sum = np.where(mask, stack, 0).sum(axis=0)
        clipped_count = mask.sum(axis=0)
        safe_count = np.where(clipped_count > 0, clipped_count, 1)
        result = np.where(
            clipped_count > 0,
            clipped_sum / safe_count,
            mean.squeeze(axis=0),
        )
        return result.astype(stack.dtype)


def get_methods() -> dict[str, Method]:
    """Return a mapping from method name to method instance."""
    methods: dict[str, Method] = {}
    for cls in (Mean, Median, SigmaClip):
        instance = cls()
        if instance.NAME in methods:
            msg = f"Duplicate averaging method: {instance.NAME}"
            raise ValueError(msg)
        methods[instance.NAME] = instance
    return methods
