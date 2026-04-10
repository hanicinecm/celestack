"""StarDetector — adaptive star detection on a single proxy frame."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl

from celestack.exceptions import StarDetectorError
from celestack.frame.core import Frame
from celestack.mask.core import Mask
from celestack.progress import progress_factory
from celestack.star_detector._detection import detect_in_segments
from celestack.star_detector._filtering import filter_stars
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
        match, then auto-estimates the optimal FWHM.

        Args:
            frame: The grayscale proxy frame (optionally dark-subtracted).
            mask: Foreground mask (same pixel dimensions as *frame*).

        Raises:
            StarDetectorError: If the frame is not grayscale or dimensions
                do not match.
        """
        if frame.array.ndim != 2:
            msg = "StarDetector requires a grayscale (2D) frame"
            raise StarDetectorError(msg)

        if frame.shape[:2] != mask.shape[:2]:
            msg = (
                f"Frame shape {frame.shape[:2]} does not match "
                f"mask shape {mask.shape[:2]}"
            )
            raise StarDetectorError(msg)

        sky_pixels = np.count_nonzero(~mask.array)
        if sky_pixels == 0:
            msg = "Mask covers the entire frame — no sky pixels to detect stars in"
            raise StarDetectorError(msg)

        self._frame = frame
        self._mask = mask
        self._fwhm: float = estimate_fwhm(frame.array, mask.array)
        self._stars: pl.DataFrame | None = None
        self._segment_labels: np.ndarray | None = None

    @property
    def fwhm(self) -> float:
        """Auto-estimated FWHM in the frame's own pixel coordinates."""
        return self._fwhm

    @property
    def stars(self) -> pl.DataFrame:
        """Detected star table.

        Spatial columns ``x``, ``y``, ``fwhm`` are in the frame's own pixel
        coordinates.  Columns ``x0``, ``y0`` hold the full-resolution
        equivalents.

        Raises:
            StarDetectorError: If :meth:`detect` has not been called yet.
        """
        if self._stars is None:
            msg = "detect() must be called before accessing stars"
            raise StarDetectorError(msg)
        return self._stars

    def detect(
        self,
        target_stars: int = 2000,
        n_segments: int = 40,
        *,
        roundness_range: tuple[float, float] = (-1.0, 1.0),
        min_separation: float | None = None,
        edge_margin: int = 5,
    ) -> pl.DataFrame:
        """Run the full adaptive detection pipeline.

        Args:
            target_stars: Desired number of output stars.
            n_segments: Number of sky segments for adaptive thresholding.
            roundness_range: (min, max) roundness bounds for DAOStarFinder.
            min_separation: Minimum distance between stars in the frame's own
                pixel coordinates.  Defaults to ``2 * fwhm``.
            edge_margin: Exclusion zone in the frame's own pixel coordinates
                around frame and mask edges.

        Returns:
            DataFrame with columns ``star_id, x, y, flux, fwhm, roundness,
            threshold, x0, y0``.  ``x``, ``y``, and ``fwhm`` are in the
            frame's own pixel coordinates; ``x0`` and ``y0`` are the
            full-resolution equivalents.

        Raises:
            ValueError: If *target_stars* < 1, *n_segments* < 1, or
                *roundness_range* is invalid.
            StarDetectorError: If no stars survive filtering.
        """
        if target_stars < 1:
            msg = "target_stars must be at least 1"
            raise ValueError(msg)
        if n_segments < 1:
            msg = "n_segments must be at least 1"
            raise ValueError(msg)
        if roundness_range[0] >= roundness_range[1]:
            msg = "roundness_range min must be less than max"
            raise ValueError(msg)

        if min_separation is None:
            min_separation = 2.0 * self.fwhm

        dsf = self._frame.downscale_factor

        # 1. Segment the sky.
        segment_labels = segment_sky(self._mask.array, n_segments)

        # 2. Compute per-segment target counts (2x proportional share).
        unique, counts = np.unique(
            segment_labels[segment_labels >= 0], return_counts=True
        )
        total_sky = int(counts.sum())
        target_per_segment = {
            int(seg): max(1, int(2 * target_stars * c / total_sky))
            for seg, c in zip(unique, counts, strict=True)
        }

        # 3. Per-segment adaptive detection with progress bar.
        bar = progress_factory(n_segments, "Detecting stars")
        try:
            raw_stars = detect_in_segments(
                self._frame.array,
                segment_labels,
                target_per_segment,
                self._fwhm,
                roundness_range,
                bar,
            )
        finally:
            bar.close()

        if len(raw_stars) == 0:
            msg = "No stars detected in any segment"
            raise StarDetectorError(msg)

        # 4. Filter (all coordinates remain in the frame's own pixel space).
        filtered = filter_stars(
            raw_stars,
            self._mask.array,
            target_stars,
            min_separation,
            edge_margin,
        )

        if len(filtered) == 0:
            msg = "No stars survived filtering"
            raise StarDetectorError(msg)

        # 5. Assign sequential star_id.
        filtered = filtered.with_row_index("star_id")

        # 6. Append full-resolution coordinate columns.
        filtered = filtered.with_columns(
            (pl.col("x") * dsf).alias("x0"),
            (pl.col("y") * dsf).alias("y0"),
        )

        self._stars = filtered
        self._segment_labels = segment_labels
        return filtered

    def plot(self, *, show_segments: bool = False) -> go.Figure:
        """Visualize detected stars and segments overlaid on the proxy frame.

        Args:
            show_segments: Whether to overlay segment boundary lines.

        Returns:
            Plotly figure with star markers and optional segment boundaries.

        Raises:
            StarDetectorError: If :meth:`detect` has not been called yet.
        """
        if self._stars is None:
            msg = "detect() must be called before plot()"
            raise StarDetectorError(msg)

        return _plot_stars(
            self._frame,
            self._stars,
            self._segment_labels,
            show_segments,
        )
