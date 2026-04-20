# Refactor Progress

Tracks implementation of [REFACTOR_PLAN.md](REFACTOR_PLAN.md). Each stage below
corresponds to one of the five staged commits described in section 11 of the
plan.

## Stages

- [x] **Stage 1 — Frame purge.** Strip `downscale_factor` from `Frame`, introduce
      a `_dtype` attribute, rewrite `bit_depth` as a dtype-backed property, drop
      `CelestackMetadata` read/write. See plan §1 and §11.4.1.
- [x] **Stage 2 — Mask split.** Remove `downscale_factor` from `Mask`; extract
      `MaskBuilder` and clustering into a new `mask_builder/` sub-package. See
      plan §2, §3 and §11.4.2.
- [x] **Stage 3 — Proxy sub-package.** Introduce `celestack.proxy` with
      `ProxyFrame`, `ProxyMask`, downscale helpers, `photutils` background
      subtraction and sidecar `metadata.toml`. See plan §4 and §11.4.3.
- [x] **Stage 4 — StarDetector migration.** Switch `StarDetector` inputs to
      `ProxyFrame`/`ProxyMask`, delete `_photometry.py`, drop `x0`/`y0`/
      `flux_local`, unify coordinates in proxy pixel space. See plan §7 and
      §11.4.4.
- [x] **Stage 5 — Config + averaging + docs cleanup.** Introduce `ProxyConfig`,
      dtype-based averaging validation, delete the dead `proxy_bit_depth`
      config, update `SPEC.md` to reflect the new architecture. See plan §5,
      §6 and §11.4.5.

## Completed Stages

### Stage 1 — Frame purge

- `Frame` no longer exposes `downscale_factor`. The attribute, the
  `downscaled_copy` method and all accompanying validation branches were
  removed. `Frame` is now a full-resolution-only container.
- Introduced `_dtype: str` (one of `"uint8"` / `"uint16"`) as the canonical
  type descriptor on `Frame`. Constructed from `FrameInfo.dtype` and validated
  on construction.
- Replaced the stored `_bit_depth` attribute with a `bit_depth` property
  derived from `_dtype` via a `{"uint8": 8, "uint16": 16}` lookup. Added a
  new public `dtype` property.
- `_validate_similarity` now checks shape + dtype only (was: shape + bit depth
  + downscale factor).
- `save_as` stops embedding Celestack metadata. Full-res Celestack-written
  TIFFs now carry only EXIF tags via `build_tiff_extratags`.
- Simplified `FrameInfo` to `(dtype: str, shape: tuple[int, ...])`. Dropped
  `CelestackMetadata`, `CELESTACK_KEY` and `parse_celestack_metadata` entirely.
- `TiffBackend.inspect` derives dtype directly from `page.dtype`.
  `PillowBackend.inspect` uses a new `MODE_DTYPE` map (replacing the previous
  `MODE_BIT_DEPTH` map).
- `plot_frame` dropped its `downscale_factor` parameter; axes are now in the
  frame's own pixel coordinates.
- `frame/_image_ops.py` was left intact except that `downscale_by_block_average`
  is no longer imported by `Frame`. It will be moved to `proxy/_downscale.py`
  in Stage 3.
- `DownscaleError` is no longer raised by the frame sub-package. The exception
  class itself is retained until Stage 5 because the old `Mask.downscaled_copy`
  still references it (removed in Stage 2).

#### Files modified

- `src/celestack/frame/core.py`
- `src/celestack/frame/_metadata.py`
- `src/celestack/frame/_backends.py`
- `src/celestack/frame/_plotting.py`

### Stage 2 — Mask split

- `Mask` no longer carries `downscale_factor`. The attribute, the
  `downscaled_copy` method, all associated majority-voting logic and the
  `DownscaleError` call site were removed. `Mask` is now a full-resolution-only
  container.
- `Mask._from_array` lost its `downscale_factor` kwarg. `save_as` no longer
  embeds any Celestack metadata — masks are written as plain 8-bit grayscale
  TIFFs.
- `plot_mask` in `mask/_plotting.py` dropped its `downscale_factor` parameter;
  axes are now in the mask's own pixel coordinates. Noise-highlight Scatter
  traces also report coordinates in native pixel space (no more scaling).
- `plot_clusters` and its helpers (`_cluster_palette`, `_downscale_nearest`,
  `_base_layout`) were moved out of `mask/_plotting.py` into the new
  `mask_builder/_plotting.py`. `mask/_plotting.py` now only contains mask
  rendering.
- Extracted the algorithmic mask builder into a dedicated `mask_builder/`
  sub-package:
  - `mask_builder/core.py` holds `MaskBuilder` (moved from
    `mask/_mask_builder.py`).
  - `mask_builder/_clustering.py` holds `ClusterWeights`, `extract_features`,
    `run_kmeans`, and `relabel_by_descending_y` (moved from
    `mask/_clustering.py`).
  - `mask_builder/_plotting.py` holds `plot_clusters`.
  - `mask_builder/__init__.py` exposes `MaskBuilder` and `ClusterWeights`.
- `mask/__init__.py` now exports `Mask` only.
- Tests were reorganized to mirror the new source layout:
  `tests/test_mask_builder/{conftest,test_core,test_clustering,test_plotting}.py`
  and a trimmed `tests/test_mask/` (core + plotting + morphology). Added
  `__init__.py` files to both test packages so pytest disambiguates module
  names. All 137 tests across `test_mask` and `test_mask_builder` pass.

#### Files modified / moved / created

- Removed: `src/celestack/mask/_mask_builder.py`,
  `src/celestack/mask/_clustering.py`.
- Modified: `src/celestack/mask/core.py`, `src/celestack/mask/_plotting.py`,
  `src/celestack/mask/__init__.py`.
- Created: `src/celestack/mask_builder/{__init__.py,core.py,_clustering.py,_plotting.py}`.
- Tests: moved `tests/test_mask/test_mask_builder.py` →
  `tests/test_mask_builder/test_core.py`; moved
  `tests/test_mask/test_clustering.py` → `tests/test_mask_builder/test_clustering.py`;
  split mask plotting out of `tests/test_mask/test_plotting.py` into
  `tests/test_mask_builder/test_plotting.py`; rewrote
  `tests/test_mask/test_core.py` and `tests/test_mask/test_plotting.py` to drop
  all `downscale_factor` assertions; added `tests/test_mask_builder/conftest.py`
  (builder fixtures) and updated `tests/test_mask/conftest.py` accordingly.

### Stage 3 — Proxy sub-package

- New `celestack.proxy` package with two types and four private helper
  modules:
  - `ProxyFrame` — downscaled, grayscale, background-subtracted `float16`
    proxy of a `Frame`. Backed by a `.npy` file and a sidecar
    `metadata.toml`. Constructor loads from disk; `from_frame` builds a
    detached proxy from a full-res `Frame` + `Mask`. Foreground pixels in
    the output are exactly `0.0`; sky pixels sit roughly zero-mean.
  - `ProxyMask` — boolean downscaled foreground mask, `.npy` + sidecar
    `metadata.toml`. `from_mask` applies majority-vote downscaling.
  - `_core.py` — sidecar helpers (`metadata_path_for`,
    `load_downscale_factor`, `write_or_verify_downscale_factor`). One
    `metadata.toml` per directory; all proxies in a directory must agree
    on `downscale_factor`.
  - `_downscale.py` — `block_average` (moved from `frame/_image_ops.py`)
    and `majority_vote` (extracted from the old `Mask.downscaled_copy`).
  - `_background.py` — thin wrapper around
    `photutils.background.Background2D`. Builds the 2D sky model over the
    unmasked region, subtracts it and zeroes masked pixels.
  - `_plotting.py` — `plot_proxy_frame`. Percentile-stretches the float16
    array using only non-zero sky pixels and renders a single PNG layout
    image.
- New `ProxyMetadataError` in `exceptions.py`.
- New `ProxyConfig` dataclass in `config.py` with
  `default_downscale_factor=2`, `background_box_size=16`,
  `background_filter_size=1`. Wired into `AppConfig` as `proxy`.
- `downscale_by_block_average` removed from `frame/_image_ops.py`
  (replaced by `proxy._downscale.block_average`).
- Tests: new `tests/test_proxy/` package (52 tests covering sidecar I/O,
  downscaling, background subtraction, proxy load/save round-trips, and
  plotting). All 290 tests outside of `test_star_detector` pass. The
  `test_star_detector` tree is expected to be broken until Stage 4 — it
  still imports the removed `CELESTACK_KEY` symbol.

#### Files modified / created

- Created: `src/celestack/proxy/{__init__.py,_core.py,_downscale.py,_background.py,_plotting.py,frame.py,mask.py}`.
- Modified: `src/celestack/config.py` (added `ProxyConfig`),
  `src/celestack/exceptions.py` (added `ProxyMetadataError`),
  `src/celestack/frame/_image_ops.py` (removed
  `downscale_by_block_average`).
- Tests: added `tests/test_proxy/{__init__.py,conftest.py,test_core.py,test_downscale.py,test_background.py,test_mask.py,test_frame.py,test_plotting.py}`.
  Trimmed `tests/test_frame/test_image_ops.py` (dropped the block-average
  tests) and fixed residual Stage-1 import errors in
  `tests/test_frame/{conftest,test_metadata,test_core}.py`.

### Stage 4 — StarDetector migration

- `StarDetector` now accepts `ProxyFrame` + `ProxyMask` (was `Frame` +
  `Mask`). Validation collapses to a shape + `downscale_factor` match
  plus the all-foreground sanity check — the grayscale / 2D guard is
  redundant because `ProxyFrame` already enforces 2D float16.
- All spatial columns (`x`, `y`, `fwhm`, `min_separation`, `edge_margin`,
  `fwhm_range`) are now in proxy pixel coordinates. The full-res
  projection (`x0`, `y0`) and all `/ downscale_factor` scaling branches
  in `detect` and `estimate_fwhm` are gone.
- `src/celestack/star_detector/_photometry.py` deleted. The
  `flux_local` column has been dropped across the codebase;
  DAOStarFinder's raw `flux` column is kept and used directly as the
  ranking key in `filter_stars`, `detect_in_segment_adaptive`, and the
  final `detect` concat/sort/cap.
- `_plotting.py` rewritten for proxy pixel coordinates: `plot_stars`
  takes a `ProxyFrame`, renders stars at `(x, y)` directly, and the
  hover tooltip drops `x0`/`y0`/`flux_local`. `_segment_boundary_coords`
  no longer takes a `downscale_factor`.
- `StarDetector.detect` casts the float16 proxy array to float32 before
  running DAOStarFinder / `scipy.ndimage` — those libraries do not
  support float16.
- `StarDetectorConfig` defaults retuned to proxy-pixel units (with the
  canonical `default_downscale_factor=2` in mind):
  - `fwhm_range` changed `(3.0, 11.0)` → `(1.5, 5.5)`.
  - `default_edge_margin` changed `15` → `8`.
  - `default_min_separation` changed `15` → `8`.
- Tests in `tests/test_star_detector/` rewritten:
  - `conftest.py` — no TIFF fixtures; detached `ProxyFrame` and
    `ProxyMask` built directly from float16/bool numpy arrays with
    synthetic Gaussian stars on a zero-mean noise floor.
  - `test_core.py` — checks the new validation branches (shape,
    downscale_factor, all-foreground), verifies no `x0`/`y0`/
    `flux_local` columns in the result.
  - `test_detection.py` / `test_filtering.py` — swap `flux_local` for
    `flux`.
- Full suite green: `uv run pytest` reports **348 passed**.

#### Files modified / deleted

- Deleted: `src/celestack/star_detector/_photometry.py`.
- Modified: `src/celestack/star_detector/core.py`,
  `src/celestack/star_detector/_detection.py`,
  `src/celestack/star_detector/_plotting.py`,
  `src/celestack/config.py` (retuned `StarDetectorConfig` defaults).
- Tests: rewrote `tests/test_star_detector/{conftest,test_core}.py`;
  updated `tests/test_star_detector/test_detection.py` and
  `tests/test_star_detector/test_filtering.py` to use `flux` instead of
  `flux_local`.

### Stage 5 — Config + averaging + docs cleanup

- `average_frames` now validates frame compatibility by `dtype` string
  (was `bit_depth` int). The reference dtype is resolved via
  `np.dtype(ref.dtype)` and applied to each band's result cast.
- `DownscaleError` removed from `src/celestack/exceptions.py`. The
  class had no remaining call sites after Stage 2.
- `ProxyConfig` (added in Stage 3) is the canonical home for the
  `default_downscale_factor`, `background_box_size`, and
  `background_filter_size` knobs. No `proxy_bit_depth` field exists
  or needs removing from `config.py` — it never made it past the spec.
- `SPEC.md` rewritten to match the new architecture:
  - Frame: no `downscale_factor`, no `downscaled_copy`, no embedded
    Celestack metadata. `dtype` is the canonical type descriptor,
    `bit_depth` is derived. Coordinates are in the frame's own pixel
    space.
  - Mask: no `downscale_factor`, no `downscaled_copy`, no embedded
    Celestack metadata. Plain 8-bit grayscale TIFF on save.
  - MaskBuilder: documented as living in `celestack.mask_builder`.
    `ClusterWeights` exported from there.
  - New "Proxy sub-package" section covering `ProxyFrame`,
    `ProxyMask`, sidecar `metadata.toml`, and the float16
    background-subtracted convention.
  - StarDetector documented as taking `ProxyFrame` + `ProxyMask`,
    returning a polars DataFrame with all coordinates in proxy pixel
    space.
  - Project structure diagram: proxies stored as `.npy` files with a
    shared `metadata.toml` sidecar per directory (not TIFFs).
  - Key Configurable Parameters table: dropped the dead
    `proxy_bit_depth` row.
- Tests: renamed `test_bit_depth_mismatch_raises` →
  `test_dtype_mismatch_raises` in `tests/test_averaging/test_core.py`
  and updated the regex match.
- Full suite green: `uv run pytest` reports **348 passed**.

#### Files modified (Stage 5)

- `src/celestack/averaging/core.py`
- `src/celestack/exceptions.py`
- `tests/test_averaging/test_core.py`
- `SPEC.md`
