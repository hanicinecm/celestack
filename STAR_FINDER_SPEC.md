# StarDetector Implementation Plan

## Context

The `StarCatalog` building block was split into `StarDetector` and `StarTracker` in the SPEC (commit `e8883bc`). This plan covers the `StarDetector` implementation — a new `star_detector` sub-package that detects stars on a single proxy frame using adaptive, segment-based detection with DAOStarFinder.

**Goal**: Given a dark-subtracted grayscale proxy frame and a foreground mask, detect a target number of stars with uniform spatial coverage across the sky region, returning a polars DataFrame with all spatial quantities in **full-resolution coordinates**.

**Coordinate system invariant**: detection runs on proxy arrays (which may have `downscale_factor > 1`), but all output coordinates, sizes, and FWHM values are immediately scaled to full-res by multiplying by `downscale_factor`. Running the same algorithm on a 2× and a 4× downscaled proxy should yield approximately the same results. Input parameters `min_separation` and `edge_margin` are expressed in full-res pixels.

---

## Package Structure

```
src/celestack/star_detector/
├── __init__.py           # exports StarDetector via __all__
├── core.py               # StarDetector class (public)
├── _fwhm.py              # FWHM auto-estimation (sweep logic)
├── _segmentation.py      # K-Means sky segmentation
├── _detection.py         # DAOStarFinder + binary search
├── _filtering.py         # Edge exclusion, proximity filter, capping
└── _plotting.py          # Star overlay + segment boundary visualization

tests/test_star_detector/
├── conftest.py           # Shared fixtures (synthetic frames, masks)
├── test_core.py          # StarDetector integration tests
├── test_fwhm.py          # FWHM estimation tests
├── test_segmentation.py  # Segmentation tests
├── test_detection.py     # Binary search tests
├── test_filtering.py     # Filtering pipeline tests
└── test_plotting.py      # Plot smoke tests
```

Also touch:
- `src/celestack/exceptions.py` — add `StarDetectorError(CelestackError)`

First implementation step (before any code): write `STAR_FINDER_SPEC.md` to the project root.

---

## Public API

### `StarDetector(frame: Frame, mask: Mask)`

**Constructor**. Stores frame and mask, then auto-estimates optimal FWHM in proxy pixels.

- Validates: frame is grayscale (2D array), mask pixel dimensions match frame (shapes must agree after accounting for any downscale factors).
- Raises `StarDetectorError` if validation fails.
- Calls `_fwhm.estimate_fwhm(frame.array, mask.array)` → returns FWHM in proxy pixels.
- Stores `self._fwhm_proxy: float` (used internally by DAOStarFinder).
- Stores `self._stars: pl.DataFrame | None = None` and `self._segment_labels: np.ndarray | None = None`.

### `detect(...) -> pl.DataFrame`

**Main detection method**.

```python
def detect(
    self,
    target_stars: int = 2000,
    n_segments: int = 40,
    *,
    roundness_range: tuple[float, float] = (-1.0, 1.0),
    min_separation: float | None = None,  # defaults to 2 * self.fwhm (full-res)
    edge_margin: int = 5,                 # in full-res pixels
) -> pl.DataFrame:
```

Pipeline:
1. Segment sky: `_segmentation.segment_sky(mask.array, n_segments)` → `segment_labels` (2D array in proxy space, sky pixels labeled 0..N-1, foreground pixels -1).
2. Per-segment adaptive detection: `_detection.detect_in_segments(frame.array, segment_labels, target_per_segment=2×proportional, fwhm=_fwhm_proxy, ...)` → raw star table with x/y in proxy coordinates.
3. Scale all spatial quantities to full-res: `x`, `y`, `fwhm_col` ×= `downscale_factor`.
4. Filter: `_filtering.filter_stars(stars, mask.array, downscale_factor, target_stars, min_separation, edge_margin)`.
5. Assign sequential `star_id`, store in `self._stars`, store `segment_labels` in `self._segment_labels`, return.

**Progress bar**: wraps the per-segment detection loop (`n_segments` steps, description `"Detecting stars"`). Created via `progress_factory(n_segments, "Detecting stars")`.

Returns DataFrame with columns: `star_id, x, y, flux, fwhm, roundness, threshold`.
All spatial columns (`x`, `y`, `fwhm`) are in full-res pixels. `threshold` is the per-segment binary-search result (in proxy image units — needed by StarTracker).

### `plot(*, show_segments: bool = False) -> go.Figure`

**Visualization**. Delegates to `_plotting.plot_stars(...)`.

- Base figure from `frame.plot()` — axes already in full-res coordinates.
- Star scatter trace: x/y are already in full-res (no additional scaling needed).
- Marker sizes and colors both proportional to `flux`.
- If `show_segments=True`: overlay segment boundary lines. Boundaries are computed from `segment_labels` and scaled by `downscale_factor` to match full-res axes.
- Raises `StarDetectorError` if `detect()` hasn't been called yet.

### Read-only properties

- `fwhm: float` — the auto-estimated FWHM in **full-res pixels** (`_fwhm_proxy * downscale_factor`).
- `stars: pl.DataFrame` — raises `StarDetectorError` if `detect()` not yet called.

---

## Private Modules

### `_fwhm.py` — FWHM Auto-Estimation

```python
def estimate_fwhm(
    image: np.ndarray,       # proxy-space grayscale array
    sky_mask: np.ndarray,    # True = foreground (same shape as image)
    *,
    n_subregions: int = 5,
    fwhm_range: tuple[float, float] = (2.0, 15.0),  # in proxy pixels
    n_fwhm_steps: int = 14,
    threshold_sigma: float = 8.0,
) -> float:                  # returns FWHM in proxy pixels
```

Algorithm:
1. Compute bounding box of sky pixels (where `sky_mask` is `False`).
2. Pick 5 sub-regions from the sky bounding box: 4 corners + center. Each sub-region is a square patch (~1/4 of the bounding box extent). Sub-regions may partially overlap the foreground; sky_mask is applied when running detection.
3. For each sub-region, sweep `np.linspace(fwhm_range[0], fwhm_range[1], n_fwhm_steps)`.
4. For each FWHM, run DAOStarFinder directly (no wrapper) at a fixed high threshold (`threshold_sigma` × background std of the sub-region's sky pixels).
5. Pick the FWHM yielding the most detections in that sub-region.
6. Return the mean optimal FWHM across all 5 sub-regions.

No progress bar — fast (5 sub-regions × 14 FWHM values, small patches).

Helper:
```python
def _pick_subregions(
    sky_mask: np.ndarray,
    n: int,
) -> list[tuple[int, int, int, int]]:
    """Return (y0, y1, x0, x1) bounding boxes for N sub-regions of the sky bbox."""
```

### `_segmentation.py` — Sky Segmentation

```python
def segment_sky(
    sky_mask: np.ndarray,
    n_segments: int,
) -> np.ndarray:
```

Algorithm:
1. Extract (row, col) coordinates of all sky pixels (where `sky_mask` is `False`).
2. Run sklearn K-Means with `n_clusters=n_segments` on the coordinate array.
3. Build a 2D label array: sky pixels get their cluster label (0..N-1), foreground pixels get -1.
4. Return the label array (proxy-space, same shape as `sky_mask`).

No progress bar — K-Means on coordinates is fast.

### `_detection.py` — DAOStarFinder + Binary Search

```python
def detect_in_segments(
    image: np.ndarray,
    segment_labels: np.ndarray,   # 2D, sky pixels labeled 0..N-1, foreground -1
    target_per_segment: dict[int, int],
    fwhm: float,                  # in proxy pixels
    roundness_range: tuple[float, float],
    progress_bar: ProgressBar,
) -> pl.DataFrame:
```

Iterates over unique non-(-1) labels in `segment_labels`. For each segment label `i`:
- Build a detection mask: `True` everywhere except pixels where `segment_labels == i` (i.e. expose only this segment to DAOStarFinder).
- Estimate background mean and std from pixels in this segment.
- Set threshold bounds: `threshold_min ≈ 2σ`, `threshold_max ≈ 15σ`.
- Call `binary_search_threshold(image, detection_mask, fwhm, target=2 × proportional_share, roundness_range, bounds)`.
- Append resulting stars with the segment label and binary-searched threshold.
- Call `progress_bar.update()`.

Return combined DataFrame with columns: `x, y, flux, fwhm_col, roundness, threshold, segment_id`. Coordinates are still in proxy space at this point — scaling to full-res happens in `StarDetector.detect()`.

```python
def binary_search_threshold(
    image: np.ndarray,
    mask: np.ndarray,             # True = ignored by DAOStarFinder
    fwhm: float,
    target_count: int,
    roundness_range: tuple[float, float],
    threshold_bounds: tuple[float, float],
    max_iterations: int = 15,
) -> tuple[float, pl.DataFrame]:
    """Binary-search threshold to yield closest to target_count stars.

    Returns (optimal_threshold, stars_df). If target is unreachable (fewer
    real stars than target), returns the lowest-threshold result within bounds.
    DAOStarFinder is called directly (no wrapper function).
    """
```

Note: `binary_search_threshold` is the reusable primitive for StarTracker propagation — it does threshold tuning only, no post-filtering.

### `_filtering.py` — Post-Detection Filtering

All filtering operates in **full-res coordinates** (caller has already scaled x/y).

```python
def filter_stars(
    stars: pl.DataFrame,
    sky_mask: np.ndarray,         # proxy-space mask
    downscale_factor: int,
    target_stars: int,
    min_separation: float,        # full-res pixels
    edge_margin: int,             # full-res pixels
) -> pl.DataFrame:
```

Pipeline (sequential):
1. **Edge exclusion**: remove stars within `edge_margin` full-res pixels of:
   - Frame edges (compare x/y against `image_shape * downscale_factor`).
   - Mask boundary (dilate `sky_mask` by 1 pixel with `scipy.ndimage.binary_dilation` to find boundary pixels; scale boundary pixel coordinates to full-res; remove stars within `edge_margin` of any boundary pixel).
2. **Proximity filter**: sort by flux descending; greedily accept stars, rejecting any candidate within `min_separation` full-res pixels of an already-accepted star.
3. **Cap**: keep the `target_stars` brightest remaining stars.

No progress bar — fast array/table operations on a small star count.

### `_plotting.py` — Visualization

```python
def plot_stars(
    frame: Frame,
    stars: pl.DataFrame,
    segment_labels: np.ndarray | None,
    show_segments: bool,
) -> go.Figure:
```

- Base: `frame.plot()` — figure already has axes in full-res coordinates.
- Stars scatter trace: x/y from `stars` DataFrame (already full-res). Marker sizes and colors from `flux` column (normalize to a reasonable visual range).
- If `show_segments=True`: compute segment boundary pixels from `segment_labels` (pixels where a neighbor belongs to a different segment or is -1), scale their coordinates by `frame.downscale_factor`, render as scatter points (small dots forming boundary lines).

---

## Error Handling

- `StarDetectorError` for: non-grayscale frame, shape mismatch between frame and mask, zero sky pixels, `plot()` or `stars` accessed before `detect()`, no stars survive filtering.
- `ValueError` for: `target_stars < 1`, `n_segments < 1`, invalid `roundness_range` (min ≥ max).

---

## Testing Strategy

### Fixtures (`conftest.py`)
- Synthetic grayscale proxy Frame (e.g. 128×128, `downscale_factor=2`) with Gaussian blobs at known positions on a noisy background.
- Simple rectangular Mask (top half sky, bottom half foreground).
- Irregular Mask (to test segmentation with non-rectangular sky).

### Key tests

**`test_fwhm.py`**:
- Gaussian blobs of known FWHM → estimated FWHM is within ±1px.
- All-foreground mask → raises `StarDetectorError`.

**`test_segmentation.py`**:
- Rectangular sky, N segments → each segment has roughly equal pixel count (within 20%).
- Foreground pixels all labeled -1.
- `n_segments=1` → entire sky is one segment.

**`test_detection.py`**:
- Binary search on synthetic image with known star count → result is within ±20% of target.
- Target higher than actual stars → returns all found without crashing.
- Target 0 → empty DataFrame.

**`test_filtering.py`**:
- Two stars within `min_separation` → fainter removed.
- Star within `edge_margin` of frame edge → removed.
- Star within `edge_margin` of mask boundary → removed.
- 100 stars, `target_stars=50` → 50 brightest returned.

**`test_core.py`**:
- Full pipeline: synthetic image with N blobs, `downscale_factor=2` → `detect()` returns ~N stars with x/y in full-res.
- Same image at `downscale_factor=4` → roughly same star positions (within pixel rounding).
- `plot()` before `detect()` → `StarDetectorError`.
- `stars` before `detect()` → `StarDetectorError`.
- `detect()` called twice → replaces previous results.

**`test_plotting.py`**:
- `plot()` → returns `go.Figure`.
- `plot(show_segments=True)` → returns `go.Figure` (smoke test).

---

## Implementation Order

1. Write `STAR_FINDER_SPEC.md` to project root (human-readable spec derived from this plan)
2. `src/celestack/exceptions.py` — add `StarDetectorError`
3. `_segmentation.py` + `test_segmentation.py`
4. `_fwhm.py` + `test_fwhm.py`
5. `_detection.py` + `test_detection.py`
6. `_filtering.py` + `test_filtering.py`
7. `_plotting.py` + `test_plotting.py`
8. `core.py` + `test_core.py`
9. `__init__.py` — exports
10. Lint/format: `ruff check --fix && ruff format`

---

## Key Files to Reference

- `src/celestack/mask/core.py` — constructor/property/plot delegation pattern
- `src/celestack/mask/_mask_builder.py` — two-step (construct + detect) API pattern
- `src/celestack/mask/_clustering.py` — sklearn K-Means usage
- `src/celestack/mask/_plotting.py` — Plotly figure patterns
- `src/celestack/frame/core.py` — `Frame.plot()`, `downscale_factor`, array access
- `src/celestack/progress.py` — `progress_factory(total, description)`
- `src/celestack/exceptions.py` — exception hierarchy

---

## Verification

1. `ruff check --fix && ruff format`
2. `pytest tests/test_star_detector/` — all new tests pass
3. `pytest` — full suite still passes
4. Manual smoke: construct `StarDetector` with a real proxy frame + mask, call `detect()`, call `plot(show_segments=True)`, visually inspect
