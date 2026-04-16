# StarTracker — Star Propagation Module

## Context

Stars are detected on the reference frame proxy by `StarDetector`, producing a polars DataFrame of ~1000 stars with sub-pixel positions, flux, FWHM, roundness, etc. The next step is **propagating** these stars across the entire frame stack to build a multi-frame correspondence table (`stars.parquet`). This table feeds `SkyTransform`, which fits the spatial transformation model.

Frames form a time series — stars move smoothly due to Earth rotation, but the motion is not analytically known. The tracker must build a motion model incrementally and use it to predict + locate each star on each frame.

## Design Overview

The tracker is organized around a single **detections tensor** and a pure **prior function** that computes predicted positions on demand from that tensor. There is no velocity cache, no per-direction bookkeeping, and no front-expansion state beyond the tensor itself.

- **`detections`**: tensor of shape `(n_stars, n_frames, 2)`.
  - `NaN` — unvisited (not yet attempted on this pass).
  - positive floats — detected sub-pixel `(x, y)` in proxy coordinates.
  - `-1` sentinel — miss (attempted, not found, or masked by foreground).

- **`calculate_prior(star, frame) → (x, y) | NaN`**: pure function of `detections`. No cached state. Called on demand for every `(star, frame)` the main loop considers.

Consequences of this shape:

- **Revisiting misses is trivial**: reset `-1 → NaN` in the desired rows, rerun the loop. The prior function naturally reflects the richer detection set from the previous pass.
- **Outlier rejection composes the same way**: flip detections back to `NaN`, rerun.
- **No middle-of-stack ambiguity**: a gap in the middle of a track is just a window with detections on both sides — interpolation, not a forward/backward dilemma.
- **Per-star decoupling**: stars are independent in the inner loop, so within-frame processing order is mostly a cache-locality concern.

## Motion Model

### Units

All pixel-valued configuration entries (`StarTrackerConfig`) are expressed in **full-resolution pixels** for parity with the rest of the project config. The tracker converts them to proxy pixels internally using the reference proxy's `downscale_factor` before running anything. Length-valued entries are divided by `downscale_factor`; area/variance-valued entries are divided by `downscale_factor²`. All pixel values below are in proxy pixels.

### Timestamp requirement

Every frame in the stack (including the reference) must have a non-`None` `timestamp`. If any frame lacks one, `StarTracker` raises `StarTrackerError` naming the offending frame.

### FWHM

Stars are point sources, so the effective FWHM is a property of the camera + optics and is assumed constant across the stack. A single FWHM value (measured once on the reference proxy during re-centroiding, or supplied via config) is used for all matched-filter convolutions and centroid windows.

### `calculate_prior(star, frame)` — tiered

The prior function tries progressively weaker estimators. The first tier that succeeds wins; lower tiers are not mixed in.

**Tier 1 — own temporal fit.** Collect this star's detections within a window of `propagation_window` frames centered on `frame` (in time, not in index — the window is truncated at stack boundaries). If ≥2 detections lie in the window, fit `x(t) = vx·t + bx`, `y(t) = vy·t + by` by unweighted least squares and evaluate at `frame.timestamp`. Also record `rms_residual` for patch sizing.

The window is a **linearity bound on the arc**, not a sample-count bound. Star trails are circular; linearity only holds within a short temporal span. Fitting across a wider window degrades accuracy as the arc bends.

Handles extrapolation on the propagation front and interpolation across interior gaps with identical logic.

**Tier 2 — identity.** Exactly 1 detection in the window, and it is at the immediately adjacent frame (in time). Use that detection's position as the prior. Variance falls back to an initial prior (`default_initial_variance`).

**Tier 3 — spatial fallback.** No own-history prior available (tiers 1–2 failed). Query the KD-tree for the K nearest stars in **reference-space** to this star's reference position, capped at `spatial_max_neighbor_distance`. For each such neighbor, recursively call `calculate_prior(neighbor, frame)` — **but restricted to Tier 1 only** (see below). Collect the neighbors for which Tier 1 succeeds; if fewer than `spatial_min_neighbors` succeed, give up.

For each contributing neighbor, compute its displacement from its reference position: `Δ_i = prior_i − ref_pos_i`. Predict this star's position as `ref_pos_star + weighted_mean(Δ_i)`, with inverse-squared-distance weights on reference-space distance to the target.

Variance: weighted sample variance of the displacements (MAD-scaled is acceptable), added to this tier's baseline uncertainty.

**Tier 4 — lost for this pass.** Return `NaN`. The main loop will skip this `(star, frame)`.

### Why Tier 3 recurses to Tier 1 only

- Tier 2 neighbor contributions are either zero-displacement (when the neighbor's single detection is the reference) or inapplicable at the target frame — they carry no useful information for the spatial estimate.
- Tier 3 recursing into Tier 3 is disallowed outright: prevents unbounded recursion and keeps the spatial tier a simple one-hop borrowing of temporal priors.

### Search radius

```text
search_radius = clamp(sqrt(var) · default_search_radius_sigma,
                      default_min_search_radius,
                      default_initial_search_radius)
```

Tier 3 priors naturally produce a wider radius because their variance is larger.

## Initialization: Re-Centroiding the Reference Frame

The `StarDetector` output is used only as an **initial guess** for where stars live on the reference frame. It uses a different sub-pixel-position algorithm (photutils DAOStarFinder-style) than the tracker's matched-filter + center-of-mass centroiding, and mixing the two centroid sources in the output would inject a systematic offset into every track's reference anchor.

**First step of `StarTracker`**: seed the tracker's own reference table by re-running the tracker's star-finding routine on the reference frame proxy at each `StarDetector`-reported position (treating it as the "predicted" position with a small search radius, e.g. `default_min_search_radius`). The tracker keeps **only** its own refined `(x, y)` and `flux_local`; the original `StarDetector` centroid and photometry are discarded. The tracker's reference rows become the sole source of truth for all downstream steps: KD-tree construction, processing order (by the tracker's own reference `flux_local`), and the anchor of every star's track.

FWHM is also measured once here (from the reference cutouts) and then held fixed for the rest of propagation.

## Data Structures

- **`detections` tensor** `(n_stars, n_frames, 2)`. Initialized to `NaN`. The reference frame column is populated with the re-centroided reference positions before propagation begins.
- **KD-tree on reference positions** (`scipy.spatial.KDTree`): built **once** at the start of propagation from every star's reference `(x, y)`. Static — never rebuilt. Used for Tier 3 neighbor queries.
- **Reference flux** vector: used only to order stars within a frame. Computed once from the re-centroided reference.

No velocity cache. No per-direction state. No miss counters (replaced by the pass-level mechanism — see below).

## Propagation Algorithm

### Frame ordering

Sort non-reference frames by ascending `|t − t_ref|`. Process in that order (interleaving forward and backward). Closest frames first = smallest motion = easiest to track with sparse priors.

### Star ordering within a frame

Descending reference `flux_local` (tiebreak on `star_id`). Fixed once after reference re-centroiding. Order matters weakly — bright stars detected early in a frame populate the detection tensor and thereby feed Tier 3 priors for fainter stars processed later in the same frame.

### Single-pass loop

```text
for frame in sorted_by_time_delta(frames, reference):
    load frame array
    for star in descending reference flux_local:
        prior = calculate_prior(star, frame)
        if prior is NaN:                        # no temporal or spatial prior — lost this pass
            continue
        if prior lands on foreground mask:
            detections[star, frame] = -1        # miss sentinel; distinguishing masked vs SNR-miss is TBD
            continue
        search in a patch sized by the prior's variance (see Star-Finding)
        if found and passes validation:
            detections[star, frame] = (x_found, y_found)
        else:
            detections[star, frame] = -1
    unload frame array
```

At pass end: every `(star, frame)` entry is either a positive position or `-1`. No `NaN` remains for any star that had at least one reachable prior during the pass.

### Multiple passes

The single-pass loop is the inner unit. Around it sits a pass-level driver:

```text
pass 1: run single-pass loop from fresh detections tensor (reference column seeded).
between passes:
    - outlier rejection: flip suspected bad detections to NaN.    [algorithm TBD]
    - miss reset:        flip -1 back to NaN (optionally filtered).
pass 2+: re-run the single-pass loop with the updated tensor.
termination / convergence criterion:                                [TBD]
```

Rationale:

- Pass 1 is mostly Tier 1 extrapolation from the reference outward, with Tier 3 rescuing the earliest-frame losses using neighbors' just-populated detections.
- Between passes, outlier rejection removes bad detections; miss reset gives previously-lost `(star, frame)` entries a fresh attempt against a now-dense tensor — interior gaps become interpolation problems rather than extrapolation ones.
- Later passes should converge quickly; termination and outlier-rejection specifics are deferred to a follow-up discussion.

### Foreground mask handling

Before attempting to find a star, check if its predicted position falls on the foreground mask. If so, record `-1`. The mask test uses the same sentinel as a detection miss in the current design — separating masked vs SNR-miss sentinels is a cheap extension to consider if re-detection policy needs to differ between them.

## Star-Finding Mechanism: Centroid Refinement

Recommended: **matched-filter peak detection + center-of-mass centroiding on cutouts**. No per-star DAOStarFinder. No pre-detection of all frames.

### Rationale

| Approach | Pro | Con |
| --- | --- | --- |
| A: Pre-detect all frames | Reuses StarDetector | Expensive upfront, wasteful, can't exploit predicted position |
| B: DAOStarFinder per star | Full shape validation | ~10ms overhead per call × 100K lookups = too slow |
| **C: Centroid refinement** | **Fast, focused, sufficient accuracy** | **No shape validation (mitigated by motion-model outlier rejection)** |

At FWHM 3–5px on 8-bit proxies, center-of-mass centroiding achieves ~0.1–0.2px accuracy — more than sufficient for building the smooth motion model that SkyTransform will fit.

### Per-star algorithm

1. **Extract cutout**: square of side `2 · search_radius` centered on predicted position. Numpy slice from the loaded frame array.
2. **Background estimation**: annulus around predicted position (reuse pattern from `compute_local_flux()` — median of pixels in 2.5–4.0 × FWHM ring). Compute `local_noise` as `1.4826 · MAD` of annulus pixels.
3. **Matched-filter peak detection**: convolve background-subtracted cutout with Gaussian kernel (σ = FWHM / 2.355). Find peak in the convolved image. If `peak < peak_min_snr · local_noise` → miss.
4. **Centroid refinement** on the **un-convolved** background-subtracted cutout:
   - Center a small window (radius ~1.5 × FWHM) on the peak from step 3.
   - Threshold: include only pixels above `max(30% of peak, 2σ of noise)`.
   - Intensity-weighted center of mass on surviving pixels.
5. **Flux measurement**: reuse `compute_local_flux()` at the refined position. Stored for downstream use; not used for validation.
6. **Validation**:
   - Reject if centroid lies outside `search_radius` of predicted position.
   - Any additional motion-model consistency checks are deferred to the outlier-rejection stage between passes (see TBD).

No per-frame flux, roundness, or shape consistency checks. The prediction model is the sole validator. If future work shows we need photometric/shape gating, it should be added at the pass-level outlier rejection stage so it can accumulate distributional statistics rather than comparing to a single-frame snapshot.

## Module Layout

```text
src/celestack/star_tracker/
    __init__.py          # exports StarTracker
    core.py              # StarTracker class (public API)
    _propagation.py      # frame ordering, pass driver, single-pass loop
    _prior.py            # calculate_prior: tiered temporal + spatial predictor
    _centroid.py         # cutout extraction, peak detection, CoM centroiding
    _plotting.py         # star track visualization
```

Tests: `tests/test_star_tracker/` mirroring source layout.

## Key Files to Modify

- `src/celestack/config.py` — add `StarTrackerConfig` to `AppConfig`
- `src/celestack/exceptions.py` — add `StarTrackerError`

## Key Files to Reuse

- `src/celestack/star_detector/_photometry.py` — `compute_local_flux()` for flux at refined positions
- `src/celestack/frame/core.py` — `Frame.load()`, `Frame.unload()`, `Frame.array`, `Frame.timestamp`
- `src/celestack/mask/core.py` — `Mask.array` for foreground checking

## Configuration

```python
@dataclass(frozen=True)
class StarTrackerConfig:
    # Temporal model (Tier 1)
    default_propagation_window: int = 5          # frames, centered on target frame (time-ordered)
    default_initial_variance: float = 64.0       # full-res px², used for Tier 2 / Tier 3 baselines

    # Spatial model (Tier 3)
    default_spatial_neighbors: int = 8
    default_spatial_min_neighbors: int = 3
    default_spatial_max_neighbor_distance_frac: float = 0.15  # fraction of the proxy's long edge

    # Search (full-res pixels; converted to proxy px internally via downscale_factor)
    default_initial_search_radius: float = 80.0  # full-res px
    default_min_search_radius: float = 12.0      # full-res px (~1 FWHM at typical sampling)
    default_search_radius_sigma: float = 4.0

    # Centroid refinement
    default_peak_min_snr: float = 3.0

    # Pass-level (TBD — placeholders; see Open Questions)
    default_max_passes: int = 3
```

## Output Schema

| Column | Type | Description |
| --- | --- | --- |
| star_id | UInt16 | From reference detection |
| frame_id | Utf8 | Filename stem (e.g. "IMG_1234") |
| x | Float32 | Proxy-pixel x |
| y | Float32 | Proxy-pixel y |
| x0 | Float32 | Full-res x |
| y0 | Float32 | Full-res y |
| flux_local | Float32 | Background-subtracted flux |

Reference frame stars included as rows where `frame_id == reference_frame_stem`. `-1` entries in the internal tensor are dropped on export.

## Verification

1. Unit tests for `_centroid.py`: synthetic star cutouts with known positions, verify centroid accuracy < 0.5px.
2. Unit tests for `_prior.py`: synthetic detection tensors, verify each tier (own-linear, identity, spatial-borrow) and the cutoffs between them.
3. Unit tests for `_propagation.py`: synthetic linear and curved tracks, verify single-pass output and the miss → reset → second-pass revival behavior.
4. Integration test: small stack of real proxy frames → verify track continuity and output schema.
5. Visual check: `StarTracker.plot()` showing star tracks overlaid on frames.

## Decisions Made

- **Detection-tensor design**: a single `(n_stars, n_frames, 2)` tensor is the sole state. Priors are a pure on-demand function of it. No velocity cache, no per-direction bookkeeping.
- **Tiered prior**: Tier 1 own-history linear fit in a time-centered window; Tier 2 identity from a single adjacent detection; Tier 3 spatial borrow of Tier-1 displacements from nearby stars. No blending across tiers — first match wins.
- **Time-centered window over fixed sample count**: the window bounds the *arc length* over which linear motion is a valid approximation; sample count is a consequence, not a constraint. Handles front extrapolation and interior interpolation identically.
- **Tier 3 recurses to Tier 1 only**: Tier 2 neighbor contributions are either zero or inapplicable; disallowing Tier 3→Tier 3 caps the recursion at one hop.
- **Multi-pass propagation**: outlier rejection and miss reset sit *between* passes. This gives previously-lost tracks a second chance against a denser tensor without complicating the inner loop.
- **Per-star decoupling inside a pass**: stars are independent; processing order is fixed to descending reference `flux_local` for determinism and weak Tier-3 enrichment within a frame.
- **Star-finding**: matched-filter peak detection + CoM centroiding. No per-star DAOStarFinder, no pre-detection.
- **Frame ID**: filename stem (string), not sequential index.
- **KD-tree is static**: built once on reference positions at the start of propagation; never rebuilt.
- **No in-loop photometric/shape validation**: the prior is the only per-frame validator. Any distributional validation belongs at the pass-level outlier rejection stage.
- **Reference re-centroiding**: the tracker re-runs its own centroiding on the reference frame before propagation so reference rows and propagated rows share a coordinate system. `StarDetector`'s centroids and `flux_local` are discarded once the tracker's refined values are written.
- **Config units**: all pixel-valued `StarTrackerConfig` entries are in full-resolution pixels; converted internally via the proxy's `downscale_factor`.

## Open Questions (deferred)

- **Outlier rejection algorithm** between passes.
- **Termination / convergence criterion** for the multi-pass driver.
- **Masked vs SNR-miss sentinels**: whether to distinguish in the detection tensor, and whether re-detection policy should differ.
