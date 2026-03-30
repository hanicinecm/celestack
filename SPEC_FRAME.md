# Frame Implementation Plan

## Context

Frame is the foundational image container for Celestack. Every other component (MaskBuilder, StarCatalog, SkyTransform, average_frames, Project) depends on it. No implementation exists yet — `src/celestack/__init__.py` is empty, `tests/` directory doesn't exist. This is the first real code in the project.

## Design Decisions

1. **Add numpy explicitly** to pyproject.toml (currently only a transitive dep).
2. **`save_downscaled` has `bit_depth=8, grayscale=True` parameters** — defaults to 8-bit grayscale output, both overridable.
3. **EXIF extraction from all formats** — exifread handles TIFF, JPEG, PNG uniformly.
4. **Tile read fallback** — non-tiled TIFFs: load full image + slice with a warning (not an error).
5. **No astropy in Frame** — reserved for later components.
6. **No filename-based timestamp inference in Frame** — Frame only reads EXIF. Batch inference is Project's responsibility.
7. **`timestamp` is `float | None`** — EXIF datetimes converted to epoch seconds. Sequence-number pseudo-timestamps are equally valid (transform fitting doesn't need real time, just a monotonically increasing proxy).
8. **Metadata stored as dict** — raw EXIF data in `_metadata: dict`. Written back to TIFF on save.
9. **`timestamp` property with override** — getter derives from `_metadata` (returns float), setter allows Project to assign pseudo-timestamps. Override takes precedence.
10. **All Celestack-written files are tiled TIFFs** — including masks (no more PNG). Enables uniform tile reading across the whole project.
11. **Celestack metadata only in proxy files** — `downscale_factor` and `timestamp` are embedded via tifffile's `metadata` parameter (JSON in ImageDescription tag) only by `save_downscaled`. Full-res files written by `save()` carry EXIF tags but no Celestack metadata. Proxies are fully self-describing; full-res frame timestamps come from EXIF, or are set via the timestamp setter before `save_downscaled` is called so they flow into the proxy.
12. **`downscale_factor` is NOT a constructor parameter** — always read from embedded Celestack metadata in the TIFF, defaulting to 1 for external files. Only `save_downscaled` produces frames with `downscale_factor > 1`.
13. **Constructor signature is `Frame(path)`** — single argument. All other state is read from the file itself (EXIF metadata, Celestack metadata, bit depth).

## Files to Create/Modify

| File | Action |
| --- | --- |
| `src/celestack/exceptions.py` | Create — custom exceptions |
| `src/celestack/frame.py` | Create — Frame class |
| `src/celestack/__init__.py` | Update — export Frame |
| `tests/conftest.py` | Create — shared test fixtures |
| `tests/test_frame.py` | Create — Frame tests |

## Two Kinds of Files

Frame transparently handles two categories of files:

1. **Files without Celestack metadata**: external user-provided files (TIFF, JPEG, PNG) and full-res Celestack files written by `save()`. `downscale_factor` defaults to 1. `timestamp` from EXIF, or `None`.
2. **Celestack proxy files** (written by `save_downscaled`): always tiled TIFF. Celestack metadata embedded in ImageDescription as JSON: `{"downscale_factor": N, "timestamp": T}`. EXIF metadata also preserved as TIFF tags.

The constructor detects which kind it's dealing with by checking for the Celestack metadata dict.

## Implementation Plan

### Step 1: Add numpy to `pyproject.toml`

Add `"numpy>=2.3.0"` to the `dependencies` list.

### Step 2: `src/celestack/exceptions.py`

```python
"""Custom exceptions for the Celestack library."""

__all__ = ["CelestackError", "DownscaleError", "TileReadError"]

class CelestackError(Exception): ...
class TileReadError(CelestackError): ...
class DownscaleError(CelestackError): ...
```

Note: `MissingTimestampError` is no longer needed — timestamp can be `None` at Frame level. Downstream components (star propagation) raise their own errors if timestamp is missing.

### Step 3: Frame constructor + metadata

**File**: `src/celestack/frame.py`

**Constants**:

```text
TIFF_SUFFIXES = frozenset({".tif", ".tiff"})
PILLOW_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
SUPPORTED_SUFFIXES = TIFF_SUFFIXES | PILLOW_SUFFIXES
DEFAULT_TILE_SIZE = 256
CELESTACK_METADATA_KEY = "celestack"
```

**`Frame.__init__(self, path: Path)`**:

1. Validate `path` exists and has a supported suffix. Store `_path = Path(path)`.
2. Read bit depth: tifffile `page.bitspersample` for TIFFs, PIL mode mapping for others. Store `_bit_depth: int`.
3. For TIFFs, check for Celestack metadata in ImageDescription tag (JSON dict). If found:
   - `_downscale_factor = celestack_meta["downscale_factor"]`
   - `_timestamp = celestack_meta.get("timestamp")` (the float override)
4. If no Celestack metadata (external file):
   - `_downscale_factor = 1`
   - `_timestamp = None` (no override — getter will try EXIF)
5. Extract EXIF metadata via `_extract_metadata()` → stored in `_metadata: dict`.

**`_extract_metadata(self) -> dict`**:

- Use `exifread.process_file(f, details=False)` for all formats.
- Build dict from available tags:
  - `"datetime"`: try `EXIF DateTimeOriginal` > `EXIF DateTimeDigitized` > `Image DateTime`. Store as raw string `"YYYY:MM:DD HH:MM:SS"`.
  - `"camera_model"`: `Image Model` tag → `str`
  - `"exposure"`: `EXIF ExposureTime` → `float` (convert Ratio)
  - `"iso"`: `EXIF ISOSpeedRatings` → `int`
  - `"focal_length"`: `EXIF FocalLength` → `float` (convert Ratio)
- Return dict with only the keys that were found.

**Public properties** (read-only except timestamp):

| Property | Type | Source |
| --- | --- | --- |
| `path` | `Path` | `_path` |
| `downscale_factor` | `int` | `_downscale_factor` (from Celestack metadata or default 1) |
| `bit_depth` | `int` | `_bit_depth` |
| `metadata` | `ExifMetadata` | `_metadata` (frozen dataclass with `datetime`, `camera_model`, `exposure`, `iso`, `focal_length`) |
| `is_tiled` | `bool` | opens TIFF and checks `page.is_tiled`; always `False` for non-TIFF files |
| `timestamp` | `float \| None` | getter/setter (see below) |

**`timestamp` property**:

```python
@property
def timestamp(self) -> float | None:
    if self._timestamp is not None:
        return self._timestamp
    dt_str = self._metadata.get("datetime")
    if dt_str is None:
        return None
    return datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S").timestamp()

@timestamp.setter
def timestamp(self, value: float) -> None:
    self._timestamp = value
```

Getter logic: return override if set (from Celestack metadata or setter), else derive from EXIF datetime string, else None.

### Step 4: Lazy array loading

**`array` as `@cached_property`**:

- TIFF: `tifffile.imread(self._path)` → numpy array with correct dtype/shape.
- Non-TIFF: `np.array(Image.open(self._path))`.
- No normalization.

**`unload(self) -> None`**:

- `del self.__dict__["array"]` (KeyError is caught silently).
- Resets cached_property so next access re-loads from disk.

### Step 5: `save(self, path: Path) -> Frame`

Used for full-res files. If the source is already a tiled TIFF, copies it verbatim; otherwise writes a new tiled TIFF with EXIF metadata — no Celestack metadata in either case. Returns a new `Frame` pointing to `path`.

1. **If `self.is_tiled`**: use `shutil.copy2(self._path, path)` and return `Frame(path)` immediately. No array loading, no re-encoding.
2. Load `self.array`.
3. Build `extratags` list from `_metadata` for EXIF preservation. tifffile supports basic TIFF tags via `extratags` tuples of `(code, dtype, count, value, writeonce)`:
   - DateTime (tag 306): written via tifffile's `datetime=` parameter from `_metadata["datetime"]`
   - Model (tag 272): `(272, 2, 0, value, True)` — ASCII string
   - ExposureTime (tag 33434): `(33434, 5, 1, (numerator, denominator), True)` — RATIONAL
   - ISOSpeedRatings (tag 34855): `(34855, 3, 1, value, True)` — SHORT
   - FocalLength (tag 37386): `(37386, 5, 1, (numerator, denominator), True)` — RATIONAL
   - Note: these are written as top-level TIFF tags (not EXIF sub-IFD, which tifffile doesn't support). Round-trip with exifread needs testing — if exifread doesn't find them, we fall back to DateTime + Model only.
4. Write with `tifffile.imwrite`:
   - `tile=(256, 256)`, `compression="zlib"`
   - `photometric="rgb"` for 3-channel, `"minisblack"` for grayscale/boolean
   - No `metadata=` argument (no Celestack JSON in ImageDescription)
5. Only include extratags for metadata keys that exist in `_metadata`.
6. Return `Frame(path)`.

### Step 6: `save_downscaled(self, path, downscale_factor, bit_depth=8, *, grayscale=True) -> Frame`

1. Guard: raise `DownscaleError` if `self._downscale_factor != 1`.
2. Guard: raise `ValueError` if `self.timestamp is None` — timestamp must be set (from EXIF or via the setter) before downscaling.
3. Load `self.array`.
4. If `grayscale=True` and array is RGB `(H, W, 3)`: convert using luminance weights `[0.2989, 0.5870, 0.1140]` via `np.dot(array[..., :3], weights)`.
5. Convert to PIL Image, resize with `Image.LANCZOS` to `(W // downscale_factor, H // downscale_factor)`. Note: PIL takes `(width, height)`.
6. Convert bit depth: scale values proportionally `(array * ((2**target - 1) / (2**source - 1)))`, cast to target dtype (8→uint8, 16→uint16, 32→float32).
7. Build Celestack metadata: `{"downscale_factor": downscale_factor, "timestamp": self.timestamp}`.
8. Save as tiled TIFF with `metadata=celestack_meta` (JSON in ImageDescription) + EXIF extratags (same tag list as `save()`).
9. Return `Frame(path)` — the new frame reads `downscale_factor` from its own embedded metadata.

### Step 7: `read_tile(self, x, y, width, height) -> np.ndarray`

1. Validate bounds: `x >= 0`, `y >= 0`, `x + width <= image_width`, `y + height <= image_height`. Raise `ValueError` on out-of-bounds.
2. If not a TIFF: raise `TileReadError`.
3. Open with `tifffile.TiffFile`, check `page.is_tiled`.
4. If tiled: use `tif.aszarr()` store for efficient slice `store[y:y+height, x:x+width]`. Close store after.
5. If NOT tiled: **fallback with warning** — `warnings.warn(...)`, load full array, slice `array[y:y+height, x:x+width]`.

### Step 8: `plot(self) -> go.Figure`

Renders the image as a Plotly layout image (background) on an empty figure whose axes are in **full-res pixel coordinates**. This is lightweight (no per-pixel trace), keeps the figure small, and makes overlaying full-res annotations straightforward.

1. Get array shape `(H, W)` (or `(H, W, 3)`). Compute full-res extents: `fw = W * self._downscale_factor`, `fh = H * self._downscale_factor`.
2. Convert the array to an 8-bit PIL Image for embedding:
   - Boolean: `(array.astype(np.uint8) * 255)`, mode `"L"`.
   - Grayscale non-boolean: scale to 8-bit `(array / array.max() * 255).astype(np.uint8)` if not already uint8, mode `"L"`.
   - RGB: scale to 8-bit per channel if needed, mode `"RGB"`.
3. Encode as base64 PNG: write PIL image to `io.BytesIO`, base64-encode, produce `"data:image/png;base64,..."` URI.
4. Build an empty `go.Figure()`.
5. Add the image via `fig.add_layout_image` with `xref="x"`, `yref="y"`, `x=0`, `y=fh`, `sizex=fw`, `sizey=fh`, `xanchor="left"`, `yanchor="top"`, `layer="below"`, `sizing="stretch"`.
6. Set axes: `fig.update_xaxes(range=[0, fw], showgrid=False)`, `fig.update_yaxes(range=[fh, 0], showgrid=False, scaleanchor="x")` (y reversed so top of image = y=fh).
7. Set `fig.update_layout(title=self._path.name)`.
8. Return `Figure` (never call `.show()`).

### Step 9: `__repr__`

```python
f"Frame({str(self._path)!r}, downscale_factor={self._downscale_factor})"
```

### Step 10: Update `src/celestack/__init__.py`

```python
"""Celestack: astrophotography image stacking library."""

__all__ = ["Frame"]

from celestack.frame import Frame
```

## Testing Strategy

### Fixtures (`tests/conftest.py`)

Small images created programmatically — no real astrophotography files needed:

| Fixture | Description |
| --- | --- |
| `tmp_rgb_tiff` | 64x48 RGB 16-bit tiled TIFF with EXIF datetime via tifffile |
| `tmp_gray_tiff` | 64x48 grayscale 16-bit tiled TIFF |
| `tmp_rgb_jpeg` | 64x48 RGB 8-bit JPEG with EXIF via PIL |
| `tmp_no_metadata_tiff` | TIFF with no EXIF data at all |
| `tmp_non_tiled_tiff` | Strip-layout (non-tiled) TIFF |
| `tmp_bool_mask_tiff` | Boolean mask saved as TIFF |
| `tmp_celestack_tiff` | TIFF with embedded Celestack metadata (downscale_factor, timestamp) |

### Test Cases (`tests/test_frame.py`)

**Constructor + metadata**:

- `test_load_rgb_tiff` — verify path, downscale_factor=1, bit_depth, camera_model from external TIFF
- `test_load_jpeg` — same from JPEG
- `test_load_celestack_tiff` — verify downscale_factor and timestamp read from embedded Celestack metadata
- `test_timestamp_from_exif` — verify float timestamp derived from EXIF datetime
- `test_no_metadata_timestamp_is_none` — TIFF without EXIF and no Celestack metadata → timestamp is None
- `test_optional_metadata_missing` — camera_model/exposure/iso/focal_length are None when absent
- `test_unsupported_suffix_raises` — `.bmp` file raises ValueError
- `test_external_file_downscale_factor_is_1` — external TIFF/JPEG/PNG always has downscale_factor=1

**Timestamp property**:

- `test_timestamp_setter_overrides_exif` — set timestamp, verify getter returns override
- `test_timestamp_setter_on_no_exif` — set timestamp on frame with no EXIF, verify getter returns it
- `test_timestamp_getter_converts_exif_to_float` — verify epoch seconds conversion is correct
- `test_celestack_timestamp_overrides_exif` — Celestack metadata timestamp takes precedence over EXIF

**Lazy loading**:

- `test_array_not_loaded_on_init` — `"array" not in frame.__dict__` after construction
- `test_array_loads_on_access` — verify numpy array with expected shape/dtype
- `test_unload_frees_array` — after unload, `"array" not in frame.__dict__`
- `test_array_reloads_after_unload` — unload then re-access, data identical

**Save**:

- `test_save_returns_frame` — returned Frame points to the new path
- `test_save_creates_tiled_tiff` — non-tiled input re-encoded as tiled TIFF
- `test_save_preserves_array_data` — save, reload, `np.array_equal`
- `test_save_preserves_exif_metadata` — save, reload, verify camera_model and datetime survive
- `test_save_does_not_embed_celestack_metadata` — save, reload, verify no Celestack metadata in ImageDescription (downscale_factor defaults to 1, timestamp comes from EXIF only)
- `test_save_copies_tiled_tiff` — already-tiled TIFF is copied verbatim; data identical, no re-encoding (verify via byte equality)

**Downscale**:

- `test_save_downscaled_dimensions` — 64x48 with factor=2 → 32x24
- `test_save_downscaled_grayscale` — input RGB, output `(H, W)` shape
- `test_save_downscaled_no_grayscale` — `grayscale=False` keeps RGB
- `test_save_downscaled_bit_depth` — 16-bit input, 8-bit target → uint8 output
- `test_save_downscaled_returns_frame` — returned Frame has correct downscale_factor read from file
- `test_save_downscaled_from_proxy_raises` — Frame with downscale_factor>1 raises DownscaleError
- `test_save_downscaled_carries_timestamp` — source timestamp (EXIF or pseudo) is preserved in output
- `test_save_downscaled_no_timestamp_raises` — frame with no timestamp (no EXIF, no setter) raises ValueError

**Tiled layout detection**:

- `test_is_tiled_tiff` — tiled TIFF returns `True`
- `test_is_tiled_non_tiled_tiff` — strip-layout TIFF returns `False`
- `test_is_tiled_non_tiff` — JPEG returns `False`

**Tile reading**:

- `test_read_tile_correct_region` — compare to full array slice
- `test_read_tile_non_tiled_warns` — non-tiled TIFF emits warning but returns correct data
- `test_read_tile_non_tiff_raises` — JPEG raises TileReadError
- `test_read_tile_out_of_bounds_raises` — ValueError on invalid coordinates

**Plot**:

- `test_plot_rgb_returns_figure` — `isinstance(result, go.Figure)`
- `test_plot_grayscale_returns_figure`
- `test_plot_boolean_mask_returns_figure`
- `test_plot_axes_in_fullres_coords` — x/y axis ranges equal `W * downscale_factor` and `H * downscale_factor`
- `test_plot_proxy_axes_in_fullres_coords` — proxy frame (downscale_factor=4) axes cover full-res extent, not proxy dimensions

## Verification

1. `ruff check --fix && ruff format` — passes clean
2. `pytest` — all tests pass
3. Manual: create a Frame from a real TIFF, inspect properties, plot it
