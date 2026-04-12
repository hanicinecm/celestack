"""StarDetector — adaptive star detection on a single proxy frame."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl

from celestack.config import CFG
from celestack.exceptions import StarDetectorError
from celestack.frame.core import Frame
from celestack.mask.core import Mask
from celestack.star_detector._fwhm import estimate_fwhm
from celestack.star_detector._plotting import plot_stars as _plot_stars
from celestack.star_detector._segmentation import segment_sky


class StarDetector:
    """Detect stars on a single dark-subtracted grayscale proxy frame.

    Construction validates the frame and mask, then auto-estimates the
    optimal FWHM.  Call :meth:`detect` to run the full adaptive detection
    pipeline, and :meth:`plot` to visualize the results.

    All spatial quantities (``x``, ``y``, ``fwhm``) in the output are in the
    **frame's own pixel coordinate system** (proxy pixels when
    ``frame.downscale_factor > 1``).  Full-resolution equivalents ``x0`` and
    ``y0`` are appended as the final columns of the returned DataFrame to align
    the stars with the full-resolution frame axes.
    """

    def __init__(self, frame: Frame, mask: Mask) -> None:
        """Prepare a star detector for the given frame and mask.

        Validates that the frame is grayscale and that the mask dimensions
        match, then initializes the internal state.

        Args:
            frame: The grayscale proxy frame (optionally dark-subtracted).
            mask: Foreground mask (same pixel dimensions as *frame*).

        Raises:
            StarDetectorError: If the frame is not grayscale or dimensions
                do not match.
        """
        # Validation:
        if frame.array.ndim != 2:
            msg = "StarDetector requires a grayscale (2D) frame"
            raise StarDetectorError(msg)

        if frame.shape[:2] != mask.shape[:2]:
            msg = (
                f"Frame shape {frame.shape[:2]} does not match "
                f"mask shape {mask.shape[:2]}"
            )
            raise StarDetectorError(msg)

        if frame.downscale_factor != mask.downscale_factor:
            msg = (
                f"Frame downscale_factor {frame.downscale_factor} does not match "
                f"mask downscale_factor {mask.downscale_factor}"
            )
            raise StarDetectorError(msg)

        if np.count_nonzero(~mask.array) == 0:
            msg = "Mask covers the entire frame — no sky pixels to detect stars in"
            raise StarDetectorError(msg)

        # Initialization:
        self._frame = frame
        self._mask = mask
        self._segment_labels: np.ndarray | None = None
        self._fwhm: float | None = None
        self._stars: pl.DataFrame | None = None

    @property
    def segment_labels(self) -> np.ndarray:
        """2D segment label array computed by :meth:`segment`.

        Sky pixels are labeled ``0..N-1``; foreground pixels are ``-1``.

        Raises:
            AttributeError: If :meth:`segment` has not been called yet.
        """
        if self._segment_labels is None:
            msg = "segment() must be called before accessing segment_labels"
            raise AttributeError(msg)
        return self._segment_labels

    @property
    def fwhm(self) -> float:
        """Auto-estimated FWHM in the frame's own pixels.

        This is the assumed stars' FWHM in the frame's own pixels, without the
        downscaling effect.  Estimated by :meth

        Raises:
            AttributeError: If FWHM has not been estimated yet.
        """
        if self._fwhm is None:
            msg = "FWHM has not been estimated yet; call detect() to estimate it"
            raise AttributeError(msg)
        return self._fwhm

    @property
    def stars(self) -> pl.DataFrame:
        """Detected star table.

        Spatial columns ``x``, ``y``, ``fwhm`` are in the frame's own pixel
        coordinates.  Columns ``x0``, ``y0`` hold the full-resolution
        equivalents.

        Raises:
            AttributeError: If :meth:`detect` has not been called yet.
        """
        if self._stars is None:
            msg = "detect() must be called before accessing stars"
            raise AttributeError(msg)
        return self._stars

    def segment(
        self,
        n_segments: int = CFG.star_detector.default_n_segments,
    ) -> None:
        """Segment the sky into regions for adaptive per-segment detection.

        Must be called before :meth:`detect`.  Can be called repeatedly with
        different values of *n_segments* to experiment; each call discards any
        previously detected stars.

        Args:
            n_segments: Number of sky segments (K-Means clusters).

        Raises:
            ValueError: If *n_segments* < 1.
        """
        if n_segments < 1:
            msg = "n_segments must be at least 1"
            raise ValueError(msg)

        # Run the k-means segmentation:
        self._segment_labels = segment_sky(self._mask.array, n_segments)
        self._stars = None  # prior detections are stale after re-segmentation

    def estimate_fwhm(
        self,
        *,
        range: tuple[float, float] = CFG.star_detector.fwhm_range,
        resolution: int = CFG.star_detector.fwhm_n_steps,
        threshold_sigma: float = CFG.star_detector.fwhm_threshold_sigma,
    ) -> None:
        """Estimate the FWHM of stars in the frame's own pixels.

        Finds the FHWM for the DAOSStarFinder algorithm, which maximizes the
        star detection for a given threshold.  The estimated FWHM is
        stored internally and used as the default for :meth:`detect`.

        Args:
            range: (min, max) FWHM in the full-resolution image pixels to sweep.
            resolution: Number of FWHM values to sweep within *range*.
            threshold_sigma: Detection threshold as a multiple of the
                background standard deviation.

        Warning: The range parameter controling the sweep is in the full-resolution
            image pixels, not the frame's own pixels.  This is to ensure similar
            behavior across different downscaling factors.
        """
        scaled_range = (
            range[0] / self._frame.downscale_factor,
            range[1] / self._frame.downscale_factor,
        )

        self._fwhm = estimate_fwhm(
            image=self._frame.array,
            segment_labels=self.segment_labels,
            fwhm_range=scaled_range,
            n_fwhm_steps=resolution,
            threshold_sigma=threshold_sigma,
        )

    # def detect(
    #     self,
    #     target_stars: int = _cfg.default_target_stars,
    #     *,
    #     max_roundness: float = _cfg.default_max_roundness,
    #     min_separation: float | None = None,
    #     edge_margin: int = _cfg.default_edge_margin,
    # ) -> pl.DataFrame:
    #     """Run the adaptive detection pipeline on the current segmentation.

    #     :meth:`segment` must be called first.

    #     Args:
    #         target_stars: Desired number of output stars.
    #         max_roundness: Maximum roundness bounds for DAOStarFinder.
    #         min_separation: Minimum distance between stars in the frame's own
    #             pixel coordinates.  Defaults to ``2 * fwhm``.
    #         edge_margin: Exclusion zone in the frame's own pixel coordinates
    #             around frame and mask edges.

    #     Returns:
    #         DataFrame with columns ``star_id, x, y, flux, fwhm, roundness,
    #         threshold, x0, y0``.  ``x``, ``y``, and ``fwhm`` are in the
    #         frame's own pixel coordinates; ``x0`` and ``y0`` are the
    #         full-resolution equivalents.

    #     Raises:
    #         StarDetectorError: If :meth:`segment` has not been called yet.
    #         ValueError: If *target_stars* < 1.
    #         StarDetectorError: If no stars survive filtering.
    #     """
    #     if self._segment_labels is None:
    #         msg = "segment() must be called before detect()"
    #         raise StarDetectorError(msg)
    #     if target_stars < 1:
    #         msg = "target_stars must be at least 1"
    #         raise ValueError(msg)

    #     if min_separation is None:
    #         min_separation = _cfg.min_separation_fwhm_multiplier * self.fwhm

    #     dsf = self._frame.downscale_factor
    #     segment_labels = self._segment_labels

    #     # 1. Compute per-segment target counts (2x proportional share).
    #     unique, counts = np.unique(
    #         segment_labels[segment_labels >= 0], return_counts=True
    #     )
    #     total_sky = int(counts.sum())
    #     target_per_segment = {
    #         int(seg): max(1, int(2 * target_stars * c / total_sky))
    #         for seg, c in zip(unique, counts, strict=True)
    #     }

    #     # 2. Per-segment adaptive detection.
    #     raw_stars = detect_in_segments(
    #         self._frame.array,
    #         segment_labels,
    #         target_per_segment,
    #         self._fwhm,
    #         (-max_roundness, max_roundness),
    #         threshold_min_sigma=_cfg.detection_threshold_min_sigma,
    #         threshold_max_sigma=_cfg.detection_threshold_max_sigma,
    #     )

    #     if len(raw_stars) == 0:
    #         msg = "No stars detected in any segment"
    #         raise StarDetectorError(msg)

    #     # 3. Filter (all coordinates remain in the frame's own pixel space).
    #     filtered = filter_stars(
    #         raw_stars,
    #         self._mask.array,
    #         target_stars,
    #         min_separation,
    #         edge_margin,
    #     )

    #     if len(filtered) == 0:
    #         msg = "No stars survived filtering"
    #         raise StarDetectorError(msg)

    #     # 4. Assign sequential star_id.
    #     filtered = filtered.with_row_index("star_id")

    #     # 5. Append full-resolution coordinate columns.
    #     filtered = filtered.with_columns(
    #         (pl.col("x") * dsf).alias("x0"),
    #         (pl.col("y") * dsf).alias("y0"),
    #     )

    #     self._stars = filtered
    #     return filtered

    def plot(self, *, show_segments: bool = False) -> go.Figure:
        """Visualize available detection results overlaid on the frame.

        Always returns a figure.  Stars are included if :meth:`detect` has
        been called; segment boundaries are included if :meth:`segment` has
        been called and *show_segments* is ``True``.

        Args:
            show_segments: Whether to overlay segment boundary lines.

        Returns:
            Plotly figure with the frame image and any available overlays.
        """
        return _plot_stars(
            frame=self._frame,
            segment_labels=self._segment_labels,
            stars=self._stars,
            show_segments=show_segments,
        )
