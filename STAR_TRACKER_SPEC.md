# StarTracker — Star Propagation Module

## Context

`StarTracker` is the second stage of Celestack's star pipeline. Its input is the reference-frame star table produced by `StarDetector` (a polars DataFrame of ~1000 stars with sub-pixel positions, flux, FWHM, roundness). Its output is `stars.parquet`: a dense `(n_stars × n_frames)` correspondence table with one row per `(star, frame)` pair. Rows where the star could not be located carry null coordinates together with a reason code (e.g. foreground-masked, below SNR, out of bounds, outlier-rejected); the exact schema is specified further down. That table is the sole input to `SkyTransform`, which fits the spatial `f(x, y, t) → (x_ref, y_ref)` model used to register the stack.

The tracker's job is **propagation**: taking the set of stars identified on the reference frame and locating each of them on every other frame in the stack. Frames form a time series — stars trace smooth, gently curving arcs across the sensor due to Earth rotation — but the motion is not known analytically and varies across the field. The tracker must therefore build a motion model incrementally from its own detections and use it to predict where each star should appear on each frame, then search a small neighborhood of that prediction with a lightweight centroiding routine. Gaps along any given star's trail are expected and normal: foreground occlusion, photon noise, cosmic rays, thin cloud, or simple SNR dropouts will all leave per-frame holes, and the tracker must tolerate them rather than treat them as failures.

All work happens on the downscaled grayscale proxy frames, consistent with the rest of the project's "analyze on proxies, output on full-res" architecture. The tracker is stack-aware but otherwise stateless across project runs: everything it needs is rebuilt from the reference table, the frame stack, and the foreground mask at the start of a run.

## Glossary

Terms below are used consistently throughout the rest of the spec. Where a term is introduced in a later section, the definition here is authoritative.

### Frames and coordinates

- **Frame** — a single proxy image in the stack: downscaled, grayscale, reduced-bit-depth, with a known `downscale_factor`. The tracker operates exclusively on frames; full-resolution originals do not appear anywhere in this spec.
- **Stack** — the ordered collection of all light-frame proxies for the project.
- **Reference frame** — the one frame whose coordinate system is the fitting target. All stars are identified here and all trails anchor here.
- **Target frame** — any non-reference frame the tracker is currently trying to locate stars on.
- **Mask** — a boolean foreground mask shared across the whole stack, aligned to the frame coordinate grid. `True` marks foreground (terrain) and `False` marks sky. Supplied to the tracker as a single input alongside the stack; the tracker does not build or modify it.
- **Coordinates** — `(x, y)` in a frame's native pixel grid. All tracker state, inputs, and outputs use these coordinates. Conversion to full-resolution pixels is the caller's concern and happens only at the project's output boundary.

### Stars and trails

- **Star** — a point source identified on the reference frame and assigned a stable `star_id`. The reference-frame detection is the canonical identity of the star; every other frame is an attempt to relocate the same physical object.
- **Trail** — the time-ordered sequence of per-frame results for a single star across the whole stack. Each frame contributes either a detection or a non-detection to the trail.
- **Trails table** — the single in-memory table holding every trail for every star, one row per `(star, frame)` pair. The tracker's primary data structure; all propagation reads and writes go through it. Serialized to `stars.parquet` on disk. In-memory shape and columns are specified later; the sync policy between the in-memory table and the parquet file is also deferred.
- **Detection** — a successful sub-pixel position measurement of a star on a given frame. Always refers to an `(x, y)` the tracker has accepted as real.
- **Non-detection** — a `(star, frame)` pair for which no coordinates are recorded. The status code captures why.
- **Status code** — the outcome category of a `(star, frame)` pair. Always populated (never null). Provisional enumeration:
  - `detected` — a valid sub-pixel position was found and accepted. Coordinates are populated.
  - `miss` — the tracker searched at the prediction and found no valid candidate (below SNR, failed validation). Coordinates null.
  - `out-of-sky` — the prediction fell on the foreground mask or outside the frame bounds, so no search was attempted. Coordinates null.
  - `no-prediction` — the motion model could not produce a prediction for this `(star, frame)`. Coordinates null.
  - `outlier` — a detection was made but later rejected by post-hoc outlier rejection. Coordinates null.

### Motion model and search

- **Motion model** — the mechanism that turns the tracker's current detections into a predicted position for any `(star, frame)` pair. Internals are specified later; the term refers only to the predictor as an abstraction. The model is not queried, and the prediction is undefined, in two cases: the star is already detected on that frame (the position is known, no prediction needed), or the frame lies beyond the model's reach given the current detection set (insufficient data — yields a `no-prediction` status).
- **Prediction** (or **predicted position**) — a point estimate `(x, y)` in frame coordinates plus an associated uncertainty, produced by the motion model for a given `(star, frame)`. Undefined (null) whenever the motion model is not queried or cannot produce one.
- **Search radius** — the radius in pixels around a prediction within which the centroiding primitive will accept a detection. Acts as the primitive's sole validation gate: any candidate detection beyond this radius from the prediction is rejected. Sized by the caller from the prediction's uncertainty and clamped to configured bounds. Also implicitly determines the primitive's patch size.

### Centroiding primitive

- **Sky array** — a float, background-subtracted, full-frame array in which foreground and out-of-frame pixels are encoded as `NaN`. Produced once per frame by the caller (typically via `photutils.Background2D`) and shared across every centroiding call on that frame.
- **Patch** (or **cutout**) — the square of pixels the centroiding primitive extracts from the sky array around a prediction. Sized internally from the search radius plus a small FWHM-scaled margin; the caller has no patch-sizing knob.

### Algorithm

- **Pass** — one full sweep of the main propagation loop over every `(star, frame)` pair under consideration. The tracker may run several passes, with inter-pass bookkeeping between them.

## The Centroiding Primitive

The centroiding primitive is the atomic unit of star-finding. Every detection in `stars.parquet` is produced by exactly one call to this primitive. The motion model, the pass driver, and outlier rejection all compose around it. Specifying its contract first anchors everything else.

### Role

Given a background-subtracted sky array (with foreground and out-of-frame pixels encoded as `NaN`), a prediction, a search radius, a FWHM, and an absolute detection threshold, the primitive either returns a validated sub-pixel detection — with every relevant `DAOStarFinder` output attached — or `None` if no acceptable star was found.

It is **stateless and pure**. It knows nothing about other frames, other stars, the motion model, passes, or the trails table. It inspects pixels in one region of one frame and returns one result.

### Signature

| Input | Type | Description |
| --- | --- | --- |
| `sky_array` | `np.ndarray` (2D float) | Background-subtracted frame in native pixels. Foreground and out-of-frame pixels are `NaN`. Produced once per frame by the caller and shared across every star on that frame. |
| `prediction` | `tuple[float, float]` | Predicted `(x, y)` in frame coordinates. |
| `search_radius_px` | `float` | Acceptance radius. A candidate detection is rejected if its centroid lies farther than this from the prediction. Also implicitly sets the patch size: `patch_radius_px = search_radius_px + 2 · fwhm_px`. |
| `fwhm_px` | `float` | Star FWHM in frame pixels, in the sense `DAOStarFinder` uses (the FWHM of the Gaussian kernel it convolves the patch with). The same value `StarDetector` uses on the reference frame is the natural choice — the primitive does not require it to match any physical PSF measurement. |
| `threshold` | `float` | Absolute detection threshold in `sky_array` flux units, passed directly to `DAOStarFinder`. Because `sky_array` is already background-subtracted, this is typically `threshold_sigma · bkg_rms` where `bkg_rms` is the caller's per-pixel noise estimate at the prediction (e.g., sampled from `Background2D.background_rms`). |

All pixel-valued parameters are in the frame's native coordinates. The primitive has no concept of downscale factors, FWHM-in-full-res-pixels, or any other project-level convention; it operates on raw pixel numbers supplied by the caller.

Return type:

```python
@dataclass(frozen=True)
class Detection:
    x: float           # DAO xcentroid, in sky_array coordinates
    y: float           # DAO ycentroid, in sky_array coordinates
    peak: float        # DAO peak: brightest pixel value of the source on sky_array
    flux: float        # DAO flux: kernel-weighted flux estimate
    sharpness: float   # DAO sharpness
    roundness1: float  # DAO roundness1 (marginal-sum symmetry)
    roundness2: float  # DAO roundness2 (Gaussian-fit residual)
    threshold: float   # threshold passed to DAOStarFinder for this call, echoed back
```

All `Detection` fields except `x`, `y`, and `threshold` are verbatim `DAOStarFinder` outputs. They are directly comparable to the reference-frame detections produced by `StarDetector` running the same algorithm on the same kind of sky data.

The primitive returns `Detection | None`. `None` means "no acceptable star was found." The caller maps `None` to the `miss` status code when writing the trails table.

### Preconditions

Short list — the primitive is deliberately permissive:

1. `prediction` lies strictly inside `sky_array` bounds. Required so the patch can be sliced (at all) around it. Not enforced internally.
2. `fwhm_px > 0`.
3. `threshold` is finite.
4. `sky_array.dtype` is a float type (`NaN` support required).

Notably **not** preconditions:

- The prediction may coincide with a `NaN` pixel. The primitive will still run; it will almost certainly return `None`. The caller is encouraged to check this and mark the `(star, frame)` as `out-of-sky` before calling, but the primitive will not raise.
- The patch may extend beyond the frame boundary or contain arbitrary amounts of `NaN`. Both are handled internally.
- `search_radius_px` is unconstrained — it may be a small fraction of `fwhm_px` or much larger; the derived patch size tracks it.

### Procedure

The actual star-finding work is delegated to `photutils.DAOStarFinder`, the same library `StarDetector` builds on. The primitive is a thin wrapper: slice a patch out of the sky array, hand it to `DAOStarFinder` with the caller's FWHM and threshold, pick the detection closest to the prediction, validate, return. Two important consequences:

- **Centroid and photometry consistency.** Reference-frame stars (from `StarDetector`) and propagated stars (from the tracker) are localized by the same `DAOStarFinder` call on the same kind of background-subtracted sky data. Every field of `Detection` — including `peak`, `flux`, `sharpness`, and the roundness statistics — lines up across the stack. No reference re-centroiding, no cross-stage photometric offsets.
- **Caller-level background subtraction.** The primitive sees only the pre-subtracted `sky_array`. Background estimation is done once per frame in the caller (typically via `photutils.Background2D`) and amortized across every star on that frame. Removes duplicate work and keeps the primitive single-purpose.

Steps:

1. **Patch extraction.** Derive `patch_radius_px = search_radius_px + 2 · fwhm_px`. Extract a square of side `ceil(2 · patch_radius_px) + 1` from `sky_array`, centered on the prediction. Pixels falling outside `sky_array` bounds are filled with `NaN`.

2. **Star detection.** Build the invalid mask `invalid = np.isnan(patch)`. Replace `NaN` with `0.0` in a copy of the patch — `DAOStarFinder`'s convolution step requires finite values, but the mask will prevent detections in those regions. Run `photutils.DAOStarFinder(fwhm=fwhm_px, threshold=threshold)(patch_finite, mask=invalid)`. If it returns no rows, return `None`.

3. **Filter detections.** Drop any returned detection whose integer-rounded `(xcentroid, ycentroid)` lands on an `invalid` pixel — the mask suppresses searches in `NaN` regions but does not fully prevent edge-effect detections from the `NaN`→0 boundary. If no detections remain, return `None`.

4. **Selection.** Among surviving detections, pick the one **closest to the prediction** in Euclidean distance. Brightness, sharpness, and roundness are *not* tiebreakers. Rationale: the prediction is our strongest independent information about which detection is the right one; in a busy patch the brightest source may be an unrelated star, hot pixel, or cosmic ray.

5. **Validation.** If `distance(centroid, prediction) > search_radius_px`, return `None`. The centroid is the selected `(xcentroid, ycentroid)` translated from patch-local into `sky_array` coordinates.

6. **Return** `Detection` populated from the selected `DAOStarFinder` row (`x`, `y`, `peak`, `flux`, `sharpness`, `roundness1`, `roundness2`) with `threshold` echoed from the input.

### Validation rule

The primitive enforces exactly one validation check:

> `distance(centroid, prediction) ≤ search_radius_px`

`search_radius_px` is the primitive's single robustness knob. A tight prediction → small radius → only very-close peaks are accepted. A loose prediction → large radius → more latitude. All other quality checks — roundness, flux consistency, shape, anomalous centroid drift — are deliberately deferred to pass-level outlier rejection, which has the distributional context to make those calls.

### Design bets

Two things the primitive deliberately does **not** protect against. They are flagged here because downstream stages inherit the responsibility for catching them:

1. **Biased detections near `NaN` edges.** `NaN` pixels are filled with zero before `DAOStarFinder` runs, which suppresses its internal Gaussian-kernel response near the `NaN` boundary and can bias centroids of half-visible stars inward. The primitive does not try to detect or reject these cases. **Outlier rejection must catch them** using motion-model residuals or shape-statistic tails — both distributional tools unavailable to the primitive.

2. **Misbinding to an unrelated nearby source.** Selection uses "nearest to prediction," which is a strong heuristic but not a guarantee. If the real star is missing (too faint, occluded, etc.) and an unrelated real source happens to fall closer to the prediction than `search_radius_px`, the primitive will return that source's centroid. Again, outlier rejection at the trail level should catch the inconsistency.

### What the primitive does NOT do

- **Background subtraction.** Caller's responsibility. `sky_array` is expected to be pre-subtracted.
- **Noise estimation.** Caller's responsibility. `threshold` is absolute.
- **Mask or bounds checking.** Invalid pixels are encoded as `NaN` in `sky_array` by the caller and routed through `DAOStarFinder`'s `mask` argument. No per-call pre-filtering in the primitive.
- **Patch sizing policy.** Derived internally from `search_radius_px + 2 · fwhm_px`. Not a caller knob.
- **Prediction.** The caller supplies the prediction and search radius. The primitive never consults the motion model.
- **Retry logic.** Returns `None`. Does not re-attempt with different parameters.
- **Shape / flux-consistency gating.** `DAOStarFinder` computes `sharpness` and the roundness statistics; the primitive returns them but does not threshold on them. Deferred to pass-level outlier rejection, which has distributional context.
- **Trails-table bookkeeping.** The caller translates `Detection` / `None` into the appropriate row update and status code.
- **Per-frame PSF or FWHM estimation.** `fwhm_px` is supplied.

### Open items

- **Miss diagnostics.** Several distinct internal reasons for `None` (no `DAOStarFinder` detections, all detections on invalid pixels, nearest detection beyond `search_radius_px`) all map to status `miss` outside. If downstream wants to distinguish them, the return type can grow to `Detection | MissReason`. Deferred.
- **Patch margin coefficient.** Hardcoded at `2 · fwhm_px`. Empirically `DAOStarFinder`'s kernel footprint is closer to `1 · fwhm_px` in radius, so the margin is conservative; tuning could shave a bit of pixel work without affecting results. Revisit only if profiling flags this.
- **Selection tiebreaker.** "Closest to prediction" ignores brightness, sharpness, and roundness once the threshold is cleared. A hybrid score is conceivable if integration testing shows the "nearest" heuristic binds to the wrong source too often — but outlier rejection is the more likely answer.
