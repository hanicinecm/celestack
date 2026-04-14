"""Configuration for the celestack app."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AveragingConfig:
    """Frozen configuration for the :mod:`celestack.averaging` sub-package.

    All fields carry sensible defaults.
    """

    default_band_height: int = 256  # rows per tiled read-band
    default_method: str = "sigma_clip"  # default combining method
    sigma_clip_kappa: float = 3.0  # sigma threshold for sigma-clipped mean


@dataclass(frozen=True)
class FrameConfig:
    """Frozen configuration for the :mod:`celestack.frame` sub-package.

    All fields carry sensible defaults.
    """

    hot_pixel_mad_sigma: float = 30.0  # MAD multiplier for hot-pixel detection


@dataclass(frozen=True)
class MaskConfig:
    """Frozen configuration for the :mod:`celestack.mask` sub-package.

    All fields carry sensible defaults.
    """

    default_noise_max_size: int = 8  # max component area for remove_noise
    default_highlight_noise_max_size: int = 128  # max noise area highlighted in plot
    plot_target_long_edge: int = 2560  # long-edge pixel target for plot downscaling


@dataclass(frozen=True)
class StarDetectorConfig:
    """Frozen configuration for :class:`~celestack.star_detector.core.StarDetector`.

    All fields carry sensible defaults.
    """

    # fwhm estimation
    fwhm_range: tuple[float, float] = (2.0, 11.0)  # FWHM limits in full-res pixels
    fwhm_n_steps: int = 49  # FWHM candidates evaluated
    fwhm_threshold_sigma: float = 2.0  # detection threshold as N * bg σ

    # detection threshold bounds
    detection_threshold_min_sigma: float = 1.0  # lower bound as N * bg σ
    detection_threshold_max_sigma: float = 10.0  # upper bound as N * bg σ

    # per-segment detect→filter refinement loop
    detection_overdetect_factor: float = 2.0  # initial over-detection multiplier

    # StarDetector method defaults
    default_n_segments: int = 40  # sky segments for segment()
    default_target_stars: int = 1000  # target star count for detect()
    default_max_roundness: float = 1.7  # DAOStarFinder bounds
    default_edge_margin: int = 15  # edges exclusion zone in full-res pixels
    default_min_separation: int = 15  # min separation in full-res pixels


@dataclass(frozen=True)
class AppConfig:
    """Frozen configuration for the celestack app.

    The configuration is hierarchical, with nested dataclasses for different components.
    """

    star_detector: StarDetectorConfig = field(default_factory=StarDetectorConfig)
    averaging: AveragingConfig = field(default_factory=AveragingConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)


def _load_config() -> AppConfig:
    """Load the app configuration.

    In the future, this could be extended to read from a file or environment variables.

    Returns:
        The default frozen configuration instance.
    """
    return AppConfig()


# Global configuration instance for the app
CFG = _load_config()
