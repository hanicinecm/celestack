# Celestack — High-Level Specification

## Overview

Celestack is an open-source Python library for stacking untracked starscape astrophotography images. Its primary goal is noise reduction through frame stacking, with correct star alignment despite the Earth's rotation. The foreground (terrain) is averaged in-place; the sky is geometrically corrected before averaging.

## Design Philosophy

- **Layer cake architecture**: pure domain logic → public API → CLI → GUI. Each layer is independently usable.
- **Project-based workflow**: a stacking session is a stateful *project* persisted to disk, allowing step-by-step execution and resumption.
- **Analysis on downscaled data, output on full-res**: all detection, registration, and transformation-fitting is done on low-resolution grayscale proxies. The fitted transformation is then applied to full-resolution originals.
- **No lock-in at the API**: the public API functions are the single source of truth reused by CLI, GUI, and direct Python usage.

## Tech Stack

| Concern | Choice |
| --- | --- |
| Language | Python (pure) |
| TIFF I/O + metadata | tifffile + exifread |
| Other image I/O | PIL (Pillow) |
| Star detection / thresholding | astropy (photutils) |
| Tabular data | polars |
| Plotting / visualization | plotly |
| CLI | Typer |
| GUI | NiceGUI |
| Async / parallel processing | asyncio + concurrent.futures where appropriate |
| Config persistence | TOML |

## Configuration

A global `celestack` config (stored in a user-level config file, e.g. `~/.config/celestack/config.toml`) controls:

- `projects_dir`: root directory under which all projects are stored
- Other possible global defaults

Projects are identified by a user-defined name. Project state lives at `{projects_dir}/{project_name}/`.

## Project Structure (on disk)

Only artifacts that are expensive to recompute or represent user decisions are persisted. Dark-subtracted proxy frames are computed in-memory as needed. Transformed full-res frames are persisted as a workflow step to enable inspection and debugging.

```text
{projects_dir}/{project_name}/
  project.toml              # project state, step status, ...
  stars.parquet             # all detected stars: (star_id, frame_id, x, y, brightness, size, roundness, ...)
  transform.*               # serialized transformation f(x, y, t) → (x_ref, y_ref)
  output.tiff               # final stacked image
  frames/
    full_res/
      light/                # copies of ingested light frames (full-res, RGB)
      dark/                 # copies of ingested dark frames (full-res, RGB)
      transformed/          # dark-subtracted, sky-corrected full-res frames (persisted)
      mask.tiff             # foreground mask (full-res, boolean)
      master_dark.tiff      # averaged dark frame (full-res)
      master_light.tiff     # averaged light frame (full-res); only if algorithmic mask is used
    proxy/
      light/                # downscaled, grayscale, reduced bit-depth proxies
      dark/                 # same for darks
      mask.tiff             # foreground mask (proxy-res, boolean, downscaled from full-res)
      master_dark.tiff      # averaged dark frame (proxy-res)
      master_light.tiff     # averaged light frame (proxy-res); only if algorithmic mask is used
```

Notes:

- The `master_light` is only created when the algorithmic mask creation path is used (needed for clustering on full RGB + full bit depth). It is not created when the user supplies an external mask loaded via `Mask(path)`.
- `stars.parquet` is a single table. Reference frame stars are the subset where `frame_id == reference_frame`. Star tracks across the stack are additional rows in the same table.
- `transform.*` is a single model that maps `(x, y, t) → (x_ref, y_ref)`, where `t` is a time-like variable relative to the reference frame (see SkyTransform section). Serialization format is owned by the SkyTransform class and depends on the model choice.

## Building Blocks

### Frame

Universal image container. Represents any single image in the system — light frames, dark frames, master darks, and master lights are all Frame instances. Masks are a separate type; see the Mask section below.

- **Constructor**: `Frame(path)`. Reads bit depth from the file. Extracts EXIF metadata (timestamp, camera model, exposure, ISO, focal length) where available; missing metadata are supported. For Celestack-written TIFFs, also reads embedded Celestack metadata (downscale factor, timestamp).
- **Lazy array access**: `frame.array` loads the image into memory on first access and caches it. `frame.load()` loads it explicitly. `frame.unload()` drops the cached array to free memory. The consistency with `frame.bit_depth` is asserted here.
- **Tile reading**: `frame.read_tile(x, y, width, height)` reads a rectangular region from disk without loading the full array. Enables memory-efficient tiled averaging of large stacks.
- **Array shape**: depends on the image — `(H, W, 3)` for full-res RGB originals, `(H, W)` for grayscale proxies. No normalization; callers handle the shape they expect.
- **Downscaling**: `frame.downscaled_copy(downscale_factor, bit_depth=8, *, grayscale=True) → Frame` returns a detached in-memory frame with the downscaled pixel data. The caller persists it via `save_as`. Only available from a downscale factor of 1. When `grayscale=True` (the default), converts RGB input to grayscale. The timestamp (including `None`) is inherited from the source frame without validation.
- **Coordinate recovery**: `downscale_factor` enables converting between proxy and full-res coordinates (multiply proxy coordinates by `downscale_factor` to get full-res coordinates).
- **Saving**: `frame.save_as(path, *, overwrite=False) → None` writes the frame to a TIFF file and binds the instance to that path, updating the backend. If the current source is already a TIFF, it is copied verbatim; otherwise the array is re-encoded. Celestack metadata (downscale factor, and timestamp if set) is embedded automatically for proxy frames (`downscale_factor > 1`). Raises `FileExistsError` if the destination exists and `overwrite=False`.
- **Dark subtraction**: `frame.subtract_dark(master_dark)`. Subtracts a master dark frame in place. Performs saturating subtraction (clamps at zero for unsigned integers), then identifies hot pixels in `master_dark` via a MAD-based threshold (pixels whose brightness exceeds median + 10 × robust_std) and replaces them in the result with the median of their valid 8-connected neighbors. Detaches `self` from its backing path. Returns `None`. `master_dark` must match in shape, bit depth, and downscale factor.
- **Plotting**: `frame.plot() → Figure` renders the image as a Plotly layout image (background) on an empty figure with axes in full-res pixel coordinates (accounting for `downscale_factor`). Lightweight — no per-pixel trace. The caller decides whether to `.show()` it or embed it.
- **Path**: `frame.path` returns the on-disk path. Raises `AttributeError` for detached (in-memory) frames. Callers are responsible for knowing whether a frame is detached; internal code uses `frame._path` directly.
- **Timestamp**: a `float | None` property. The getter returns (in priority order): an explicitly set override, a float derived from EXIF datetime, or `None`. The setter allows external code (e.g. Project) to assign pseudo-timestamps. Proxy frames embed the timestamp (if any) in Celestack metadata via `save_as`; it is the caller's responsibility to ensure the timestamp is set before persisting proxies that require it.
- **Downscale factor**: read-only, derived from Celestack metadata embedded in proxy TIFFs. Defaults to 1 for all other files (external files and full-res Celestack files). Only `downscaled_copy` produces frames with a downscale factor greater than 1.
- **Two kinds of files**: Frame transparently handles external files (user-provided TIFF/JPEG/PNG with EXIF metadata, downscale factor defaults to 1) and Celestack proxy files (tiled TIFFs with embedded Celestack metadata — fully self-describing). Full-res files written by Celestack to the project are TIFFs with EXIF but no Celestack metadata, and are treated the same as external files by the Frame class.

### Mask

Boolean foreground mask. The core type of the `mask` sub-package — a lightweight, file-backed container for a single-channel boolean image. Masks are distinct from `Frame` — they carry no EXIF metadata and no timestamp.

- **Constructor**: `Mask(path)`. Loads from any supported image format (TIFF, JPEG, PNG). Any non-zero pixel becomes foreground (`True`). For Celestack-written proxy masks, also reads embedded Celestack metadata (downscale factor).
- **Lazy array access**: `mask.array` loads and booleanizes the image on first access, caching the result as a `bool` array of shape `(H, W)`. `mask.unload()` drops the cache to free memory.
- **Saving**: `mask.save_as(path, *, overwrite=False) → None` writes the boolean array to an 8-bit grayscale TIFF (`True` → 255, `False` → 0) and binds the instance to that path. Always writes from the boolean array — never copies verbatim — to guarantee a clean on-disk format. Proxy masks (`downscale_factor > 1`) have Celestack metadata embedded automatically. Raises `FileExistsError` if the destination exists and `overwrite=False`.
- **Downscaling**: `mask.downscaled_copy(downscale_factor) → Mask` returns a detached in-memory mask using majority voting (blocks where more than half the pixels are foreground become foreground). The caller persists it via `save_as`, which embeds `downscale_factor` in Celestack metadata automatically. No timestamp is required or stored.
- **Auto-cleaning small particles**: `mask.remove_noise(max_size=8) → None` removes tiny connected components in both phases using 4-connectivity: small foreground particles are cleared to background, and small background holes are filled to foreground.
- **Manual editing**: `mask.mask_rectangle(x0, y0, x1, y1, *, foreground)` and `mask.mask_pixels(pixels, *, foreground)` mutate the boolean array in place and detach the mask from its backing path. Coordinates are in the mask's native pixel space. These are fire-and-forget operations with no edit recording — available on any `Mask` instance regardless of how it was created.
- **Plotting**: `mask.plot() → Figure` renders the boolean mask as a black-and-white PNG on a Plotly figure with axes in full-resolution pixel coordinates (accounting for `downscale_factor`). Lightweight — no per-pixel trace. The caller decides whether to `.show()` it or embed it.
- **Path**: `mask.path` returns the on-disk path. Raises `AttributeError` for detached (in-memory) masks.
- **Downscale factor**: read-only, from embedded Celestack metadata. Defaults to 1 for full-resolution and external masks.
- **In-memory construction**: `Mask._from_array(array)` creates a mask from a boolean NumPy array with no backing path. Used internally by `MaskBuilder.build()`.

### MaskBuilder

Builds a foreground mask algorithmically from an RGB source image via K-Means clustering. The builder is a tool specifically for clustering — it produces an approximate mask from cluster labels. Any further manual editing (rectangle/pixel masking) is done on the `Mask` object returned by `build()`.

For the **external mask** path, the project caller loads the user-supplied file directly as `Mask(path)` — no `MaskBuilder` is involved.

**Full-resolution workflow**: All clustering operates on the full-resolution pixel data. Clustering results are not invariant to bit depth and scaling changes, so downscaled proxies cannot be used for trial clustering. The constructor takes only the source RGB `Frame` and immediately extracts a standardized feature matrix (R, G, B standardized to zero mean / unit variance; X, Y normalized to [-1, 1]). This matrix is computed once and reused across all subsequent `compute_clusters` calls.

**`ClusterWeights` dataclass**: Controls which feature dimensions are included in clustering and how strongly each is weighted. Fields: `r`, `g`, `b` (RGB channel weights, default `1.0`) and `x`, `y` (normalized spatial coordinate weights, default `0.0`). Setting a weight to `0` excludes that feature dimension from the feature matrix entirely. Weights are applied to the pre-standardized features, so they act as a clean relative importance knob. Default weights use RGB only (no spatial). Exported publicly from `celestack.mask`.

**`compute_clusters(n_clusters, weights=ClusterWeights())`**: Applies weights to the stored standardized features, drops zero-weighted columns, and runs K-Means. Can be called repeatedly with different K or weights without recomputing the base features.

**Plotting**: `plot_clusters()` returns a Plotly figure with a static color-coded PNG of the cluster map and a non-interactive legend showing cluster indices. Large label arrays are downscaled via nearest-neighbor before PNG encoding (target long edge ~2560 px) to keep rendering fast; axes always reflect full-resolution dimensions. Does not embed per-pixel Plotly traces — all pixel data is serialized as a PNG data URI.

**`build(foreground_clusters)`**: Builds the boolean mask from the current cluster labels. Takes a set of cluster indices to mark as foreground. Returns a detached in-memory full-resolution `Mask`.

The algorithmic path requires interactive user input (choosing cluster count and foreground labels). To preserve the principle that API methods are the single source of truth for both CLI and GUI, mask building is decomposed into multiple atomic, non-interactive methods on Project (e.g. compute clusters, build mask). Each method takes concrete inputs and produces concrete outputs. The interactive loop — presenting results and collecting user choices — lives entirely in the CLI/GUI layer, which orchestrates these atomic methods. Manual mask refinement (rectangle/pixel edits) is done on the `Mask` instance after `build()`.

### StarCatalog

Owns star detection, propagation, and the `stars.parquet` table.

- **Reference star detection**: operates on the (optionally dark-subtracted) reference frame proxy (sky region only, masked). Auto-segments the sky into N segments, adapts thresholding per segment to achieve uniform spatial coverage of a target star count.
- **Star propagation**: propagates detected stars bidirectionally across the stack, ordered by time delta from the reference frame. Uses linear extrapolation from K nearest known positions, with outlier rejection (local linear fit, shape consistency, thresholding bounds).
- **Data**: single `stars.parquet` table with columns `(star_id, frame_id, x, y, brightness, size, roundness, ...)`. Reference frame stars are the subset where `frame_id == reference_frame`. Grows as the stars are propagated across the stack.

### SkyTransform

Fits and evaluates the spatial transformation model.

- **Fitting**: takes the full star correspondence table from StarCatalog and fits a single function `f(x, y, t) → (x_ref, y_ref)`, where `t` is a time-like variable relative to the reference frame. `t` does not need to be real acquisition time — any monotonically increasing proxy is sufficient (e.g. EXIF timestamps converted to epoch seconds, or auto-incremented sequence numbers extracted from filenames). All frames are assumed to be from one continuous session, so the transform varies smoothly with `t`.
- **Model**: non-parametric (no assumed analytical form). Model type TBD — this is a key open design decision that will affect fitting strategy, outlier detection, serialization format, and runtime performance.
- **Outlier detection**: after fitting, flags stars with large residuals. If a large fraction of a frame's stars are outliers, the frame is flagged for exclusion.
- **Evaluation**: given a pixel `(x, y)` and time `t`, returns `(x_ref, y_ref)`.
- **Application**: `sky_transform.apply(frame, mask, t) → Frame` — takes a dark-subtracted Frame, the foreground mask, and its acquisition time offset. Remaps sky pixels (where mask is False) to the reference grid; foreground pixels (where mask is True) are left in place. Returns a new Frame.
- **Serialization**: persisted as `transform.*` in the project root. The SkyTransform class owns its own `save()`/`load()` — file format depends on the model choice.

### `average_frames` (function)

Standalone utility function for averaging a list of Frames into a single Frame. Reused for master dark, master light, and the final stack.

- **Signature**: `average_frames(frames: list[Frame], method: str = "sigma_clip", *, reference_frame: Frame | None = None, band_height: int | None = DEFAULT_BAND_HEIGHT) → Frame`
- **Methods**: configurable, mean, median, sigma-clipped mean, ...
- **Output**: returns a new detached (in-memory) `Frame` with the averaged pixel data. Metadata is inherited from `reference_frame`, or from the first input frame when `reference_frame` is `None`. The caller persists the result via `frame.save_as(path)`.
- **Memory efficiency**: uses tiled reading (`frame.read_tile()`) to process the stack in horizontal row-bands so that only a small slice of each frame is held in memory at a time. Band height is an internal tunable.
- **Ingest note**: all Celestack-written files are TIFF files inheriting some of the original exif data. Proxy files additionally embed Celestack metadata and are fully self-describing.

### Project

Top-level orchestrator and the single public API. Owns the workflow state, delegates all heavy computation to domain classes.

**State** (persisted in `project.toml`):

- Celestack version that created the project
- Light frame names and dark frame names (darks may be empty)
- Reference frame name
- Per-step status: `not started` or `done`
- Project config: proxy parameters (`downscale_factor`, `bit_depth`), stacking parameters (averaging function, sigma clip kappa, etc.)

Project config is a passive collection of parameters — changing a config value does not by itself alter any project artifacts. Changes only take effect when the workflow step that consumes the parameter is re-run, which triggers the standard invalidation cascade on all downstream steps. This means any parameter can be freely changed at any time; the user simply re-runs the relevant step to apply the new value.

**Frame management**: paths are derived by convention from frame names and the project directory structure. Frame instances are constructed on-the-fly when needed via `Frame(path)`. For proxy frames, all metadata (downscale factor, timestamp) is read from embedded Celestack metadata. For full-res frames, EXIF metadata is read from the file; pseudo-timestamps for frames without EXIF are set on Frame instances via the timestamp setter before saving the proxies.

**Ingest**: copies original files into the project directory as TIFFs, generates proxies via `Frame.save_downscaled()`, and handles timestamps. When copying a full-res file, if the source is already a TIFF, Project uses `shutil.copy2` instead of re-encoding through the array. Non-TIFF inputs (JPEG, PNG) are re-written via `frame.save()`. EXIF timestamps are extracted automatically; for frames without EXIF timestamps, Project performs batch inference on the full set of filenames (stripping common prefixes/suffixes and checking whether the remaining part forms a monotonically increasing sequence usable as a time-like variable). Inferred pseudo-timestamps are persisted in the proxy Frame instances via the timestamp setter before use. Lights and darks are ingested separately; darks are optional. Accepted input formats: TIFF and common image formats supported by Pillow (JPEG, PNG, etc.). RAW camera formats (CR2, NEF, ARW, ...) are not supported. After ingest, the project is self-contained.

**Select reference frame**: records the user's choice of reference frame in `project.toml`. All star detection and transform fitting is relative to this frame.

**Workflow methods**: each workflow step (ingest, build master dark, detect stars, etc.) is a method on Project. Each method has a corresponding status entry (`not started` / `done`) recorded in `project.toml`. Each method is decorated with its prerequisites. The decorator handles:

- **Pre-check**: verifies all prerequisite steps are `done`; raises an error if not.
- **Invalidation confirmation**: on re-run of a completed step, computes the list of downstream steps that would be invalidated. If non-empty, calls `Project.on_invalidation(step_name, downstream) → bool`. If the callback returns `False`, the step aborts with an exception. If `on_invalidation` is `None` (the default), proceeds silently — safe for scripting and tests.
- **Invalidation cascade**: resets each downstream step to `not started` by calling its invalidation counterpart.
- **Post-success**: marks the step as `done` in `project.toml`.

**`on_invalidation` callback**: `Project.__init__` accepts an optional `on_invalidation` parameter with signature `(step_name: str, downstream: list[str]) → bool`. Each layer supplies its own: the CLI prompts in the terminal, the GUI opens a confirmation dialog, and direct API callers pass their own or leave the default.

**Invalidation counterparts**: each workflow method has a paired private method that cleans up its artifacts (e.g. deletes `stars.parquet`, clears `frames/full_res/transformed/`). Called automatically by the cascade, never by the user.

**Plotting methods**: read-only methods that return Plotly `Figure` objects from the current project state. Fully separated from workflow methods. Example: `plot_stars(frame_id) → Figure` renders detected star positions overlaid on the given frame's proxy image.

## Workflow Steps and Dependency Graph

Each step has a set of prerequisite steps declared via a decorator. Re-running a completed step invalidates all downstream steps (resets them to `not started` and cleans up their artifacts).

```text
ingest lights ─────────────> select reference frame
ingest darks (optional) ─┐          │
    └─> average master dark ─┐      │
                             v      v
                        define foreground mask
                            └─> detect reference stars (on proxy)
                                    └─> propagate stars (across stack of proxies)
                                            └─> fit transform
                                                    └─> apply transform (on full-res, persist)
                                                            └─> average final image
```

"Select reference frame" depends only on "ingest lights." When darks are ingested, "average master dark" is a prerequisite of "define foreground mask" — the first step that may need dark-subtracted frames — and transitively of all downstream steps. When no darks are ingested, dark subtraction is simply skipped. Dark subtraction is applied implicitly by the Project wherever needed (mask building, star detection, apply transform); it is not a standalone workflow step.

Steps are individually callable. If prerequisites are not met, the step raises an informative error rather than silently failing.

## Public API

The `Project` class is the single public API. All workflow steps are methods on `Project`. Visualization is fully separated from computation.

Design principles:

- **Workflow methods** perform computation and mutate project state.
- **Plotting methods** read project state and return Plotly `Figure` objects. They do not mutate state. The caller decides what to do with the figure (`.show()` in REPL, embed in NiceGUI, etc.).

## CLI Layer

- Built with **Typer**
- Each workflow method maps to a CLI subcommand
- Project is referenced by name: e.g. `celestack --project milkyway_june stack --func median`
- Commands that produce visualizations open a browser with the Plotly figure
- Step invalidation warnings are shown before destructive re-runs

## GUI Layer

- Built with **NiceGUI**
- Mirrors the same public API functions
- Provides interactive tools for steps requiring user input:
  - Mask definition: Plotly lasso/rectangle selection rendered in NiceGUI
  - Cluster labeling: cluster overlay with foreground/background toggle per cluster
- Shows step status (`done` / `not started`) as a workflow progress panel
- Embeds Plotly figures inline

## Key Project Configurable Parameters (non-exhaustive)

| Parameter | Description |
| --- | --- |
| `proxy_downscale_factor` | Factor by which to downscale full-res frames to produce proxies (e.g. 4 means 1/4 linear dimensions, i.e. 1/16 the pixel count) |
| `proxy_bit_depth` | Bit depth of proxy images |
| `target_stars` | Number of stars to detect in reference frame |
| `sky_segments` | Number of sky segments for adaptive thresholding |
| `propagation_window` | Number of frames used for linear extrapolation |
| `roundness_min` | Minimum star roundness to accept detection |
| `stacking_func` | Averaging function for the final stack |
| `dark_stacking_func` | Averaging function for master dark |
| `sigma_clip_kappa` | Sigma threshold for sigma-clipped mean averaging |
