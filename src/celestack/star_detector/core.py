"""StarDetector — adaptive star detection on a single proxy frame."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import polars as pl

from celestack.config import CFG
from celestack.exceptions import StarDetectorError
from celestack.progress import progress_factory
from celestack.proxy.frame import ProxyFrame
from celestack.proxy.mask import ProxyMask
from celestack.star_detector._detection import detect_in_segment_adaptive
from celestack.star_detector._fwhm import estimate_fwhm
from celestack.star_detector._plotting import plot_stars as _plot_stars
from celestack.star_detector._segmentation import (
    proportional_targets,
    segment_bfs_tree,
    segment_sky,
)


class StarDetector:
    """Detect stars on a single background-subtracted proxy frame.

    Construction validates the proxy frame and mask, then :meth:`detect`
    runs the adaptive per-segment detection pipeline.  :meth:`plot`
    visualizes the results.

    All spatial quantities (``x``, ``y``, ``fwhm``) in the output are in
    proxy pixel coordinates.
    """

    def __init__(self, frame: ProxyFrame, mask: ProxyMask) -> None:
        """Prepare a star detector for the given proxy frame and mask.

        Validates that the frame and mask share the same shape and
        ``downscale_factor`` and that at least one sky pixel is available.

        Args:
            frame: Background-subtracted float16 proxy frame.
            mask: Boolean proxy foreground mask.

        Raises:
            StarDetectorError: If shapes or downscale factors do not match,
                or if the mask covers every pixel.
        """
        if frame.shape != mask.shape:
            msg = f"Frame shape {frame.shape} does not match mask shape {mask.shape}"
            raise StarDetectorError(msg)

        if frame.downscale_factor != mask.downscale_factor:
            msg = (
                f"Frame downscale_factor {frame.downscale_factor} does not match "
                f"mask downscale_factor {mask.downscale_factor}"
            )
            raise StarDetectorError(msg)

        if not np.any(~mask.array):
            msg = "Mask covers the entire frame — no sky pixels to detect stars in"
            raise StarDetectorError(msg)

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
        """Auto-estimated FWHM in proxy pixels.

        Raises:
            AttributeError: If FWHM has not been estimated yet.
        """
        if self._fwhm is None:
            msg = "FWHM has not been estimated yet; call estimate_fwhm() first"
            raise AttributeError(msg)
        return self._fwhm

    @property
    def stars(self) -> pl.DataFrame:
        """Detected star table in proxy pixel coordinates.

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
    ) -> np.ndarray:
        """Segment the sky into regions for adaptive per-segment detection.

        Must be called before :meth:`detect`.  Can be called repeatedly with
        different values of *n_segments* to experiment; each call discards any
        previously detected stars.

        Args:
            n_segments: Number of sky segments (K-Means clusters).

        Returns:
            The 2D segment label array (same value as :attr:`segment_labels`).

        Raises:
            ValueError: If *n_segments* < 1.
        """
        if n_segments < 1:
            msg = "n_segments must be at least 1"
            raise ValueError(msg)

        self._segment_labels = segment_sky(self._mask.array, n_segments)
        self._stars = None
        return self._segment_labels

    def estimate_fwhm(
        self,
        *,
        range: tuple[float, float] = CFG.star_detector.fwhm_range,
        resolution: int = CFG.star_detector.fwhm_n_steps,
        threshold_sigma: float = CFG.star_detector.fwhm_threshold_sigma,
    ) -> float:
        """Estimate the FWHM of stars in proxy pixels.

        Finds the FWHM for the DAOStarFinder algorithm that maximises the
        star detection count at a high threshold.  The estimated value is
        stored internally and used as the default for :meth:`detect`.

        Args:
            range: (min, max) FWHM in proxy pixels to sweep.
            resolution: Number of FWHM values to sweep within *range*.
            threshold_sigma: Detection threshold as a multiple of the
                background standard deviation.

        Returns:
            The estimated FWHM in proxy pixels (same value as
            :attr:`fwhm`).
        """
        self._fwhm = estimate_fwhm(
            image=self._frame.array,
            segment_labels=self.segment_labels,
            fwhm_range=range,
            n_fwhm_steps=resolution,
            threshold_sigma=threshold_sigma,
        )
        return self._fwhm

    def detect(
        self,
        target_stars: int = CFG.star_detector.default_target_stars,
        *,
        max_roundness: float = CFG.star_detector.default_max_roundness,
        min_separation: float = CFG.star_detector.default_min_separation,
        edge_margin: int = CFG.star_detector.default_edge_margin,
        fwhm: float | None = None,
    ) -> pl.DataFrame:
        """Run the adaptive detection pipeline on the current segmentation.

        :meth:`segment` must be called first.  If *fwhm* is not supplied,
        :attr:`fwhm` must have been set by :meth:`estimate_fwhm`.

        Args:
            target_stars: Desired number of output stars.
            max_roundness: Maximum roundness bounds for DAOStarFinder.
            min_separation: Minimum distance between stars in proxy pixels.
            edge_margin: Exclusion zone in proxy pixels around frame and
                mask edges.
            fwhm: Optional FWHM override in proxy pixels.  When ``None``
                (the default), falls back to the previously estimated
                :attr:`fwhm`.

        Returns:
            DataFrame with columns ``star_id, x, y, flux, fwhm, roundness,
            threshold, threshold_sigma, segment_id``.  All spatial columns
            are in proxy pixel coordinates.  ``flux`` is the raw
            DAOStarFinder flux used to rank stars.  ``threshold`` is the
            absolute DAOStarFinder threshold; ``threshold_sigma`` is the
            same value as a multiple of background σ.

        Raises:
            StarDetectorError: If :meth:`segment` has not been called yet.
            ValueError: If *target_stars* < 1.
            StarDetectorError: If no stars survive filtering.
        """
        if self._segment_labels is None:
            msg = "segment() must be called before detect()"
            raise StarDetectorError(msg)
        if target_stars < 1:
            msg = "target_stars must be at least 1"
            raise ValueError(msg)

        effective_fwhm = fwhm if fwhm is not None else self.fwhm
        segment_labels = self._segment_labels
        image = self._frame.array.astype(np.float32, copy=False)
        sky_mask = self._mask.array
        roundness_range = (-max_roundness, max_roundness)
        cfg = CFG.star_detector

        # 1. Per-segment proportional targets, with +1 slack per segment so
        # the final cap can land on exactly *target_stars* despite rounding
        # and filter shortfalls.
        target_per_segment = proportional_targets(segment_labels, target_stars)
        target_per_segment = {seg: t + 1 for seg, t in target_per_segment.items()}

        # 2. Per-segment adaptive detect→filter.  BFS over segment adjacency
        # so each child can seed its binary search with its parent's final
        # threshold (more stars ⇒ lower threshold).
        bfs_order = segment_bfs_tree(segment_labels, (0.0, 0.0))
        per_segment: list[pl.DataFrame] = []
        seg_threshold_sigma: dict[int, float] = {}

        with progress_factory(len(target_per_segment), "Detecting stars") as bar:
            for seg_id, parent_id in bfs_order:
                initial_sigma = (
                    seg_threshold_sigma.get(parent_id)
                    if parent_id is not None
                    else None
                )
                best = detect_in_segment_adaptive(
                    image,
                    segment_labels,
                    seg_id,
                    sky_mask,
                    target_per_segment[seg_id],
                    effective_fwhm,
                    roundness_range,
                    min_separation=min_separation,
                    edge_margin=edge_margin,
                    threshold_min_sigma=cfg.detection_threshold_min_sigma,
                    threshold_max_sigma=cfg.detection_threshold_max_sigma,
                    overdetect_factor=cfg.detection_overdetect_factor,
                    initial_threshold_sigma=initial_sigma,
                )
                if best is not None:
                    per_segment.append(best)
                    seg_threshold_sigma[seg_id] = float(best["threshold_sigma"][0])
                bar.update()

        if not per_segment:
            msg = "No stars survived filtering in any segment"
            raise StarDetectorError(msg)

        stars = pl.concat(per_segment).sort("flux", descending=True).head(target_stars)
        stars = stars.with_row_index("star_id").with_columns(
            pl.col("star_id").cast(pl.UInt16)
        )

        self._stars = stars
        return stars

    def plot(self, *, show_segments: bool = False) -> go.Figure:
        """Visualize available detection results overlaid on the proxy frame.

        Always returns a figure.  Stars are included if :meth:`detect` has
        been called; segment boundaries are included if :meth:`segment` has
        been called and *show_segments* is ``True``.

        Args:
            show_segments: Whether to overlay segment boundary lines.

        Returns:
            Plotly figure with the proxy image and any available overlays.
        """
        return _plot_stars(
            frame=self._frame,
            segment_labels=self._segment_labels,
            stars=self._stars,
            show_segments=show_segments,
        )
