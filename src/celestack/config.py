"""Configuration for the celestack app."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StarDetectorConfig:
    """Frozen configuration for :class:`~celestack.star_detector.core.StarDetector`.

    All fields carry sensible defaults.
    """

    # fwhm estimation
    fwhm_range: tuple[float, float] = (2.0, 15.0)  # FWHM limits in full-res pixels
    fwhm_n_steps: int = 27  # FWHM candidates evaluated
    fwhm_threshold_sigma: float = 4.0  # detection threshold as N * bg σ

    # detection threshold bounds
    detection_threshold_min_sigma: float = 2.0  # lower bound as N * bg σ
    detection_threshold_max_sigma: float = 15.0  # upper bound as N * bg σ

    # StarDetector method defaults
    default_n_segments: int = 40  # sky segments for segment()
    default_target_stars: int = 2000  # target star count for detect()
    default_max_roundness: float = 1.7  # DAOStarFinder bounds
    default_edge_margin: int = 20  # edges exclusion zone in full-res pixels
    default_min_separation: int = 20  # min separation in full-res pixels


@dataclass(frozen=True)
class AppConfig:
    """Frozen configuration for the celestack app.

    The configuration is hierarchical, with nested dataclasses for different components.
    """

    star_detector: StarDetectorConfig = field(default_factory=StarDetectorConfig)


def _load_config() -> AppConfig:
    """Load the app configuration.

    In the future, this could be extended to read from a file or environment variables.

    Returns:
        The default frozen configuration instance.
    """
    return AppConfig()


# Global configuration instance for the app
CFG = _load_config()
