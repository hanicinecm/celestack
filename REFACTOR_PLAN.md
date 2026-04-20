# Celestack Refactor — Implementation Plan

## Context

`SPEC.md` describes the current architecture; `REFACTOR.md` describes a rewrite motivated by pain discovered while speccing the StarTracker (propagation). The root cause is that proxy frames currently inherit the uint8/uint16 dtype of the source, retain sky gradients, and are conflated with full-res Frame semantics (downscale_factor baked into Frame, TIFF metadata round-trips, lazy backends, tile readers, etc.). This makes propagation logic fragile — stars don't sit cleanly on a zero-centered noise floor.

The fix: standardize proxies as background-subtracted float arrays with stars rising from ~0 noise. Move the downscale concept out of `Frame`/`Mask` entirely and into a dedicated `proxy` sub-package that stores plain `.npy` files with sidecar `metadata.toml`. `Frame` and `Mask` become simple full-res types.

The downstream consumers (`StarDetector`, averaging, masking) are trimmed: no more dual-coordinate system (proxy vs full-res pixels), no photometry helper, all defaults in proxy space.

---

## High-level Target Layout

```text
src/celestack/
  __init__.py
  config.py
  exceptions.py
  progress.py
  frame/                 # simplified — full-res only
    __init__.py
    core.py              # Frame (no downscale_factor)
    _backends.py         # TIFF + Pillow
    _image_ops.py        # dark subtraction, bad-pixel interp, bit depth conversion, to_grayscale
    _metadata.py         # EXIF only (CelestackMetadata removed)
    _plotting.py         # plot_frame (no downscale_factor)
  mask/                  # Mask class only
    __init__.py
    core.py              # Mask (no downscale_factor)
    _morphology.py       # unchanged
    _plotting.py         # plot_mask (no downscale_factor)
  mask_builder/          # NEW — extracted from mask/
    __init__.py
    core.py              # MaskBuilder
    _clustering.py       # ClusterWeights + extract_features + run_kmeans
    _plotting.py         # plot_clusters
  proxy/                 # NEW
    __init__.py
    _core.py             # shared helpers (load/save .npy + metadata.toml)
    frame.py             # ProxyFrame
    mask.py              # ProxyMask
    _downscale.py        # block-average (from frame/_image_ops), majority-vote (from mask/core)
    _background.py       # photutils Background2D wrapper
    _plotting.py         # plot_proxy_frame (float) — reuses mask plot where useful
  averaging/             # unchanged surface; dtype handling updated
    __init__.py
    core.py
    _methods.py
  star_detector/         # trimmed
    __init__.py
    core.py              # StarDetector — acts on ProxyFrame + ProxyMask
    _detection.py        # thresholds now in float proxy units
    _fwhm.py             # unchanged except units
    _segmentation.py     # unchanged
    _plotting.py         # drop downscale_factor arithmetic
    # _photometry.py DELETED
```

---

## 1. `celestack.frame` sub-package

### `frame/core.py` — Frame class

Changes:
- **Remove** `_downscale_factor`, `downscale_factor` property, `downscaled_copy()`, `DownscaleError` usages in this module.
- **Remove** `_validate_similarity` check on downscale_factor; keep shape + dtype checks.
- **Add** `_dtype: str` attribute, set in `__init__` from `FrameInfo.dtype` (see metadata change below). Validate `_dtype in {"uint8", "uint16"}` on construction; otherwise raise.
- **Replace** `bit_depth` attribute with a property: `return {"uint8": 8, "uint16": 16}[self._dtype]`.
- **Add** `dtype` property returning `self._dtype`.
- **Keep** `path`, `shape`, `metadata`, `timestamp` (getter+setter), `array` (lazy), `load()`, `unload()`, `save_as()`, `subtract_dark()`, `interpolate_bad_pixels()`, `read_tile()`, `plot()`.
- **In `save_as`**: stop embedding CelestackMetadata (downscale_factor, timestamp). Timestamp becomes a runtime override only; full-res Celestack-written TIFFs carry EXIF as usual. (See Q4 — may need to keep timestamp embedding.)
- Update `_detached_copy` to not pass downscale_factor.

### `frame/_backends.py`

- `TiffBackend.inspect()` returns `FrameInfo(dtype, shape)` — drop celestack_metadata field.
- `PillowBackend.inspect()` returns `FrameInfo(dtype, shape)`; derive dtype from PIL mode.
- `is_tiff_path`, `get_backends` unchanged.

### `frame/_image_ops.py`

- Keep: `scaled_preview_uint8`, `convert_bit_depth`, `to_grayscale`, `interpolate_bad_pixels`, `build_dark_bad_pixel_mask`, `subtract_master_dark`.
- Remove: `downscale_by_block_average` — move to `proxy/_downscale.py`.

### `frame/_metadata.py`

- Keep: `ExifMetadata`, `extract_exif_metadata`, `as_float`, `parse_datetime_to_epoch`, `to_rational`.
- Simplify `FrameInfo`: `dtype: str`, `shape: tuple[int, ...]`. (Drop `bit_depth` field; drop `celestack_metadata` field.)
- Remove: `CelestackMetadata`, `CELESTACK_KEY`, `build_tiff_extratags`, `parse_celestack_metadata` (unless we keep timestamp embedding — see Q4).

### `frame/_plotting.py`

- `plot_frame(array, *, bit_depth, title, show_pixels)` — drop `downscale_factor` param; axes are just array pixel coordinates.

---

## 2. `celestack.mask` sub-package (Mask only)

### `mask/core.py`

- **Remove** `_downscale_factor`, `downscale_factor` property, `downscaled_copy()`, downscale-related metadata writes in `save_as()`, `DownscaleError`.
- `_from_array(array)` — drop `downscale_factor` kwarg.
- `save_as` — write plain 8-bit TIFF, no Celestack metadata.
- `plot()` — no downscale_factor.

### `mask/_morphology.py`

- Unchanged.

### `mask/_plotting.py`

- `plot_mask(array, *, title, highlight_noise)` — drop `downscale_factor`.

### Relocated out of `mask/`

- `_mask_builder.py` → `mask_builder/core.py`
- `_clustering.py` → `mask_builder/_clustering.py`
- `plot_clusters` → `mask_builder/_plotting.py`

### `mask/__init__.py`

- Export `Mask` only. Drop `MaskBuilder`, `ClusterWeights`.

---

## 3. `celestack.mask_builder` sub-package (NEW)

- `core.py` — `MaskBuilder` class. Unchanged logic; still operates on full-res RGB `Frame` (which is now the only flavor of Frame). Returns full-res `Mask`.
- `_clustering.py` — `ClusterWeights`, `extract_features`, `run_kmeans`, `relabel_by_descending_y`.
- `_plotting.py` — `plot_clusters`.
- `__init__.py` — export `MaskBuilder`, `ClusterWeights`.

Import sites to update:
- `tests/test_mask/test_mask_builder.py` → `tests/test_mask_builder/`
- Any SPEC.md references (to be updated at the end).

---

## 4. `celestack.proxy` sub-package (NEW)

### `proxy/_core.py` (shared helpers)

- `METADATA_FILENAME = "metadata.toml"`
- `_metadata_path_for(npy_path: Path) -> Path` — returns `npy_path.parent / METADATA_FILENAME`.
- `load_downscale_factor(npy_path) -> int` — reads sidecar TOML, raises `ProxyMetadataError` if missing / malformed / missing key / non-int.
- `write_or_verify_downscale_factor(npy_path, downscale_factor) -> None` — if sidecar missing, write it; if present, verify value matches, else raise.
- New exception: `ProxyMetadataError` (in `exceptions.py`).

TOML schema (single key):

```toml
downscale_factor = 4
```

### `proxy/_downscale.py`

- `block_average(array: np.ndarray, factor: int) -> np.ndarray` — moved verbatim from `frame/_image_ops.py::downscale_by_block_average`.
- `majority_vote(mask: np.ndarray, factor: int) -> np.ndarray` — the boolean downscaling logic currently inside `Mask.downscaled_copy` (threshold 0.5).

### `proxy/_background.py`

- `subtract_background(array_float: np.ndarray, mask_bool: np.ndarray, *, box_size: int, filter_size: int) -> np.ndarray`
  - Build `photutils.background.Background2D(array_float, box_size=box_size, filter_size=filter_size, mask=mask_bool)`.
  - Subtract `bkg.background.astype(np.float16)`.
  - Zero out masked (foreground) region.
  - Return the float16 result.

### `proxy/mask.py` — `ProxyMask`

```python
class ProxyMask:
    def __init__(self, path: Path):
        """Load a boolean .npy with sidecar metadata.toml."""
    @classmethod
    def from_mask(cls, mask: Mask, downscale_factor: int) -> ProxyMask: ...
    @property
    def path(self) -> Path: ...         # raises if detached
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def downscale_factor(self) -> int: ...
    @property
    def dtype(self) -> str: ...         # always "bool"
    @property
    def array(self) -> np.ndarray: ...  # bool
    def save_as(self, path: Path, *, overwrite: bool = False) -> None: ...
```

Rules:
- Constructor: `np.load(path)`; assert bool dtype; load downscale_factor from sidecar.
- `from_mask`: call `majority_vote(mask.array, factor)`; instance has `_path = None`.
- `save_as`: `np.save(path, self._array)`; `write_or_verify_downscale_factor(path, self.downscale_factor)`.
- No lazy loading, no unload, no plotting, no in-place editing.

### `proxy/frame.py` — `ProxyFrame`

```python
class ProxyFrame:
    def __init__(self, path: Path):
        """Load a float16 .npy with sidecar metadata.toml."""
    @classmethod
    def from_frame(
        cls,
        frame: Frame,
        mask: Mask,
        downscale_factor: int,
        *,
        box_size: int = CFG.proxy.background_box_size,
        filter_size: int = CFG.proxy.background_filter_size,
    ) -> ProxyFrame: ...
    @property
    def path(self) -> Path: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def downscale_factor(self) -> int: ...
    @property
    def dtype(self) -> str: ...         # always "float16"
    @property
    def array(self) -> np.ndarray: ...  # float16, centered near 0, stars < 1, foreground == 0
    def save_as(self, path: Path, *, overwrite: bool = False) -> None: ...
    def plot(self) -> Figure: ...
```

`from_frame` implementation (order matters):

1. `src_dtype = frame.dtype` (must be uint8/uint16).
2. `gray = to_grayscale(frame.array)` — preserves dtype. (For shape `(H, W, 3)` → `(H, W)`; for 2D input, just use as-is.) Importantly this runs on the full-res array.
3. Block-average: `small = block_average(gray, downscale_factor)` — still integer dtype.
4. Convert to float16 and rescale to `[0, 1]` using `np.iinfo(src_dtype).min` and `max`.
5. Build proxy mask: `pm = ProxyMask.from_mask(mask, downscale_factor)`.
6. `subtract_background(float_array, pm.array, box_size=..., filter_size=...)`.
7. Return `ProxyFrame(_path=None, _array=<result>, _downscale_factor=downscale_factor)`.

Plot: render as a float16 heatmap/image. Because foreground is 0.0 and sky noise is roughly zero-mean, display should use a stretch (e.g. `vmin = np.percentile(sky, 1)`, `vmax = np.percentile(sky, 99)`). Keep it similar in spirit to `plot_frame` (single image layer, not per-pixel traces). See Q5.

### `proxy/__init__.py`

- Export `ProxyFrame`, `ProxyMask`.

---

## 5. `celestack.config`

Changes in `config.py`:

- **Add** `ProxyConfig` dataclass:
  - `background_box_size: int = 16`
  - `background_filter_size: int = 1`
  - (Proxy dtype is fixed float16, not configurable.)
- **Remove** `proxy_bit_depth` from anywhere it appears (it's dead — proxies are always float16).
- **Update `StarDetectorConfig`** — re-express the "full-res pixel" defaults in proxy pixels (see Q8 for exact values). Values affected:
  - `fwhm_range: tuple[float, float]` — currently `(3.0, 11.0)` full-res.
  - `default_edge_margin: int` — currently 15 full-res.
  - `default_min_separation: int` — currently 15 full-res.
- **Wire** `ProxyConfig` into `AppConfig`.

---

## 6. `celestack.averaging`

- `average_frames(frames, method, *, reference_frame=None, band_height=...)`:
  - **Validate** shape + `dtype` match (replacing the current shape + bit_depth check).
  - Metadata inheritance unchanged (no more downscale_factor to preserve).
  - Continue using `frame.read_tile(...)`.
  - Result is a detached Frame with the reference frame's dtype. Ensure this still constructs correctly after removing downscale_factor from Frame.
- No change to the method implementations (`Mean`, `Median`, `SigmaClip`).

---

## 7. `celestack.star_detector`

- **Delete** `_photometry.py`. Drop `compute_local_flux` calls in `_detection.py::detect_in_segment`. DAOStarFinder already provides a raw flux; keep that column under the name `flux` (rename the dataframe's current `flux` → the kept value, drop `flux_local`). Adjust `_plotting.py::_build_star_hover_lines` to stop referencing `flux_local`.
- **StarDetector constructor** now takes `ProxyFrame` + `ProxyMask` (not `Frame` + `Mask`).
  - Validate matching `shape` and matching `downscale_factor`.
  - Validate the proxy is 2D (grayscale) — guaranteed by ProxyFrame, so this collapses to a simple shape check.
  - Validate non-zero sky pixels (`not proxy_mask.array.all()`).
- **Coordinate system unification**: drop the full-res projection. Remove `x0`, `y0` columns from the stars DataFrame. All coordinates (`x`, `y`, `fwhm`) live in proxy pixel space.
- **Config units**: `fwhm_range`, `edge_margin`, `min_separation` are now proxy pixels. Remove the internal `/ downscale_factor` scaling in `estimate_fwhm`, `detect`.
- **Thresholds**: threshold_min_sigma / threshold_max_sigma / overdetect_factor are unitless — no change, but verify that MAD-based σ on background-subtracted float16 still works (see Q7).
- **Plotting**: `plot_stars` — axes in proxy pixel coordinates (no `downscale_factor` arithmetic). `_segment_boundary_coords` — drop `downscale_factor` arg.

---

## 8. Exceptions

- Remove `DownscaleError` (no longer raised anywhere).
- Add `ProxyMetadataError(CelestackError)`.
- Keep `CelestackError`, `TileReadError` (if still used elsewhere), `MaskError`, `StarDetectorError`.

---

## 9. Tests restructuring

Mirror the new src tree:

```text
tests/
  conftest.py, utils.py
  test_frame/
    conftest.py
    test_core.py           # drop downscale tests; add dtype guard tests
    test_image_ops.py      # block_average moves to test_proxy
    test_metadata.py       # drop CelestackMetadata tests
    test_plotting.py       # drop downscale_factor param
  test_mask/
    conftest.py
    test_core.py           # drop downscale_copy tests
    test_morphology.py
    test_plotting.py       # drop downscale_factor
  test_mask_builder/       # NEW (move from tests/test_mask/test_mask_builder.py + test_clustering.py)
    conftest.py
    test_core.py
    test_clustering.py
    test_plotting.py
  test_proxy/              # NEW
    conftest.py
    test_core.py           # metadata.toml read/write/verify
    test_downscale.py      # block_average + majority_vote
    test_background.py     # subtract_background
    test_frame.py          # ProxyFrame — from_frame + save/load round-trip
    test_mask.py           # ProxyMask — from_mask + save/load round-trip
    test_plotting.py
  test_star_detector/
    # drop test_photometry? (none currently exists)
    conftest.py            # fixtures now build ProxyFrame + ProxyMask
    test_core.py           # updated coord system, no x0/y0
    test_detection.py
    test_fwhm.py
    test_filtering.py
    test_segmentation.py
    test_plotting.py
  test_averaging/
    conftest.py
    test_core.py           # update bit_depth→dtype assertion
```

Test-utility updates:
- `tests/test_mask/conftest.py` — factory fixtures for Mask without downscale_factor; move MaskBuilder fixtures to `test_mask_builder/conftest.py`.
- `tests/test_star_detector/conftest.py` — fixtures return a ProxyFrame built from a synthetic Frame+Mask, not a downscaled Frame.
- `tests/test_proxy/conftest.py` — small synthetic RGB Frame + matching Mask fixtures; tiny downscale factors (2, 4).

---

## 10. Files to touch (summary for a junior dev)

Delete:
- `src/celestack/star_detector/_photometry.py`
- `tests/test_mask/test_mask_builder.py` (relocated)
- `tests/test_mask/test_clustering.py` (relocated)

Rename / move:
- `src/celestack/mask/_mask_builder.py` → `src/celestack/mask_builder/core.py`
- `src/celestack/mask/_clustering.py` → `src/celestack/mask_builder/_clustering.py`
- `plot_clusters` logic from `src/celestack/mask/_plotting.py` → `src/celestack/mask_builder/_plotting.py`

Create:
- `src/celestack/proxy/{__init__.py,_core.py,frame.py,mask.py,_downscale.py,_background.py,_plotting.py}`
- `src/celestack/mask_builder/{__init__.py,core.py,_clustering.py,_plotting.py}`
- `tests/test_proxy/*`
- `tests/test_mask_builder/*`

Modify heavily:
- `src/celestack/frame/core.py` (remove downscale_factor, replace bit_depth with dtype-backed property)
- `src/celestack/frame/_metadata.py` (simplify `FrameInfo`; drop CelestackMetadata)
- `src/celestack/frame/_backends.py` (dtype in FrameInfo)
- `src/celestack/frame/_image_ops.py` (move block_average out)
- `src/celestack/frame/_plotting.py` (drop downscale_factor)
- `src/celestack/mask/core.py` (remove downscale_factor, downscaled_copy, metadata embedding)
- `src/celestack/mask/_plotting.py` (drop downscale_factor)
- `src/celestack/mask/__init__.py` (expose Mask only)
- `src/celestack/star_detector/core.py` (accept ProxyFrame/ProxyMask; drop x0/y0; drop unit conversion)
- `src/celestack/star_detector/_detection.py` (drop compute_local_flux call; drop flux_local)
- `src/celestack/star_detector/_fwhm.py` (units)
- `src/celestack/star_detector/_plotting.py` (drop downscale_factor arithmetic)
- `src/celestack/averaging/core.py` (dtype-based validation)
- `src/celestack/config.py` (add ProxyConfig; re-scale StarDetectorConfig defaults)
- `src/celestack/exceptions.py` (drop DownscaleError; add ProxyMetadataError)

---

## 11. Decisions locked in (from user Q&A)

1. **Precision**: `float16`, as REFACTOR.md proposes. Accept the half-precision loss for 16-bit sources as an explicit cost trade-off.
2. **`metadata.toml`**: one per directory — all `.npy` files in the same folder share one sidecar. All proxies in a folder must therefore share the same `downscale_factor`.
3. **Timestamps**: only on full-res `Frame` (via EXIF or pseudo-timestamp override). `ProxyFrame` carries no timestamp; sidecar TOML stores only `downscale_factor`. StarTracker/SkyTransform will read `t` from the full-res frame via name convention (future Project concern).
4. **PR scope**: staged. Land in ~5 commits in this order:
   1. **Frame purge** — strip `downscale_factor`, introduce `_dtype`, rewrite `bit_depth` as a property, drop `CelestackMetadata` read/write.
   2. **Mask split** — remove `downscale_factor` from `Mask`; extract `MaskBuilder` and clustering into new `mask_builder/` sub-package.
   3. **Proxy sub-package** — introduce `celestack.proxy` with `ProxyFrame`, `ProxyMask`, downscale helpers, background subtraction, sidecar `metadata.toml`.
   4. **StarDetector migration** — switch inputs to `ProxyFrame`/`ProxyMask`, drop `_photometry.py`, drop `x0`/`y0`/`flux_local`, unify coords in proxy space, update unit-dependent configs.
   5. **Config + averaging + docs cleanup** — new `ProxyConfig`, dtype-based averaging validation, delete dead `proxy_bit_depth`, update `SPEC.md` to match.
5. **Dark subtraction**: *out of scope* for this refactor. It will be a responsibility of the (not-yet-implemented) Project orchestrator: at ingest, the orchestrator loads darks, builds a master dark, subtracts it and interpolates hot pixels into each light, and persists the dark-corrected full-res frames. By the time `ProxyFrame.from_frame` is called, the `Frame` is already dark-corrected (or not, if no darks were supplied). ProxyFrame does not care either way.
6. **ProxyFrame plotting**: percentile stretch — use `vmin=np.percentile(sky_pixels, 1)`, `vmax=np.percentile(sky_pixels, 99)` computed over sky pixels only (i.e. mask out foreground zeros when computing percentiles). Render via the same single-image-layer approach as `plot_frame` (no per-pixel traces).
7. **StarDetector defaults**: divide current full-res defaults by 2 (canonical downscale_factor assumed to be 2). Concretely:
   - `fwhm_range = (1.5, 5.5)` (was `(3.0, 11.0)` full-res).
   - `default_edge_margin = 8` (was 15 full-res, rounded from 7.5).
   - `default_min_separation = 8` (was 15, rounded from 7.5).
   - Add `default_downscale_factor: int = 2` to the new `ProxyConfig` dataclass.
8. **Hot-pixel interpolation**: lives only on `Frame.interpolate_bad_pixels(master_dark)`. `ProxyFrame` is agnostic to whether its source Frame was bad-pixel-corrected or not — it simply does not touch hot pixels.
9. **Proxy `bit_depth` config**: dead. Remove any reference to `proxy_bit_depth` in SPEC.md and configs. Proxies are always float16.
10. **MAD-σ thresholds in float space**: keep existing unitless defaults (`detection_threshold_min_sigma`, `detection_threshold_max_sigma`, `overdetect_factor`) unchanged. MAD is still meaningful on background-subtracted float16 proxies. Revisit only if star-count convergence breaks during verification.

---

## 12. Verification

After implementation:

- `uv run ruff check --fix && uv run ruff format`
- `uv run pytest` — entire suite green.
- Manual smoke test script (not committed):
  1. Load a small RGB TIFF as `Frame`.
  2. Build a `Mask` via `MaskBuilder` with trivial K=2 clusters.
  3. `pf = ProxyFrame.from_frame(frame, mask, downscale_factor=4)`.
  4. `pf.save_as(tmp / "p.npy")`; reload via `ProxyFrame(tmp / "p.npy")`; assert `array.dtype == float16`, `downscale_factor == 4`.
  5. Run `StarDetector(pf, pm).detect(target_stars=50)` and assert a non-empty table with columns in proxy pixel space.
  6. Assert: sky pixel median ≈ 0; foreground pixels == 0.0 exactly.
  7. `pf.plot().show()` — visually inspect.
- Grep post-checks:
  - `grep -R "downscale_factor" src/celestack/frame src/celestack/mask` → no hits.
  - `grep -R "DownscaleError\|compute_local_flux\|CelestackMetadata" src/` → no hits.
  - `grep -R "flux_local\|x0\|y0" src/celestack/star_detector` → no hits.
