"""Default configuration for the star_detector sub-package.

All algorithm tuning knobs are defined here so they are visible and
changeable in one place.  Call :func:`get_config` to retrieve the active
configuration.

Values that are purely internal implementation details (random seeds,
sub-sampling counts, visual settings) live as module-level constants in
their respective private modules and are not exposed here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StarDetectorConfig:
    """Frozen configuration for :class:`~celestack.star_detector.core.StarDetector`.

    All fields carry sensible defaults.  Create a new instance with keyword
    overrides to experiment with different settings.
    """

    # fwhm estimation
    fwhm_range: tuple[float, float] = (2.0, 15.0)  # FWHM limits in image pixels
    fwhm_n_steps: int = 27  # FWHM candidates evaluated
    fwhm_threshold_sigma: float = 4.0  # detection threshold as N * bg σ

    # detection threshold bounds
    detection_threshold_min_sigma: float = 2.0  # lower bound as N * bg σ
    detection_threshold_max_sigma: float = 15.0  # upper bound as N * bg σ

    # StarDetector method defaults
    default_n_segments: int = 40  # sky segments for segment()
    default_target_stars: int = 2000  # target star count for detect()
    default_max_roundness: float = 1.7  # DAOStarFinder bounds
    default_edge_margin: int = 5  # exclusion zone around edges (px)
    min_separation_fwhm_multiplier: float = 2.0  # in multiples of FWHM


def get_config() -> StarDetectorConfig:
    """Return the active :class:`StarDetectorConfig`.

    Returns:
        The default frozen configuration instance.
    """
    return StarDetectorConfig()
