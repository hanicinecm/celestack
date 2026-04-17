# Refactoring Specification

I have encountered some problems down the line all the way in specing the propagation and the StarTracker building block and I have decided to bite the bullet and completely change how the frames work.
The primary motivation for this complete rewrite is to standardize the proxy frames into background-subtracted float arrays, where each star grows as a peak from a zero-centered noise.
And as this touches very lowest building blocks of the project, a complete rewrie is appropriate.

## The `celestack.frame` sub-package

- The `frame.Frame` class will change considerably.
- The Frame will only be used to load in the full-res photos
- Consequently, the downscale factor is purged from the Frame instance completely. Frames do not have downscale factors. They do, however, still encode the timestamp, and all the exif data.
- The frame instance should save `dtype: str` also, as well as the bit depth and it should raise if it's not one of `{"uint8", "uint16"}`. This is just a guard, as it's assumed elsewhere. The datatype should come from the `FrameInfo` dataclass.
- Consequently, drop the bit depth as a separate attribute. Just use the `_dtype` in the `bit_depth` property.
- As far as I can tell, the refactoring of the `Frame` class should be straight-forward - just purging the downscale factor from everywhere, and completely changing/replacing the `downscaled_copy` method.
- The `downscaled_copy` method is is gone completely. It will be replaced by the new `celestack.proxy.ProxyFrame.from_frame` class method.
- All the other modules of the `frame` sub-package should be adapted to not recognize the downscale factor at all. Any code used currently for downscaling a frame will be moved to a new `proxy` sub-package, to support the `ProxyFrame.from_frame` class method.

## The `celestack.mask` sub-package

### The `Mask` class

- The `mask` package will newly *only contain* the `Mask` class and it's related code and helpers.
- Analogous changes as to the `Frame` - the `Mask` does not know about downscaling at all.
- Every `Mask` is a full-res mask.
- There is no method for saving downscaled masks. This will be replaced by the new `celestack.proxy.ProxyMask.from_mask` class method.
- Other than that, leave it as it is.
- The what-is-now `mask` sub-package should be renamed to `mask_builder` and only contain the mask builder code (code turning `Frame` to `Mask`)

### The `MaskBuilder` class

- The `MaskBuilder` class will move from `mask` sub-package to a dedicated `mask_builder` sub-package, together with all its code.
- The `MaskBuilder` always creates `Mask` from a `Frame`, with everything that comes with the changes described above - always full-res from full-res.
- All the code currently in `mask` class related only to `MaskBuilder` moves with it to `mask_builder` sub-package.

## The new `celestack.proxy` sub-package

- The sub-package exposes two core classes - `ProxyFrame` and `ProxyMask`.
- Both are the same as `Frame` and `Mask` but they do define the downscale factor.
- Both have initializer that loads it from a path to a `.npy` file.
- The initializer loads the array directly as it was saved by the `save_as` method (see below).
- The initializer will read the downscale factor from the `metadata.toml` which is located next to the path (same dir as the npy file). Or raises if it's not there or does not containt the correct key or it's not an integer.
- Both have the `from_frame` and `from_mask` class methods. Ordinarily, these will be used to first create them, and after they are saved, they will be loaded via the initializer.
- We don't need to bother with the lazy loading and with loading/unloading.
- Also no attached/detached state. The proxies will never be manipulated with. They will be created, dumped, loaded again. The `_path` is either `Path` or `None`, the `path` property either raises or returns `Path`.
- When calling `save_as`, next to dumping their array to the `.npy` path, the method will see if the downscale factor metadata is there, it will write it if not, and it will raise, if it's there and not matching the downscale factor on the instance.
- No backends - never read from an image file. No allowed suffixes, always `.npy`.
- No cache conserving (cause no cache...)
- Probably no detached copies are needed for either of the two.
- Both have the `dtype` property which reads the dtype directly from the array, no special attribute.

### The `ProxyMask` class

- Like `Mask` but much simpler.
- Always just bool, either built from the Mask class or loaded from a bool npy file.
- No manipulation - the additional masking of pixels, rectangles, noise - all of that is done on `Mask`. The proxy is just a wrapper around downscaled numpy array with the factor in state.
- No plotting. Not needed.
- The `from_mask(cls, mask: Mask, downscale_factor: int) -> ProxyMask` class method will just downscale the `Mask.array` (use existing logic from current `mask` sub-package) and save it as a boolean array to an instance without a `_path`.

### The `ProxyFrame` class

- Like `Frame` but much simpler.
- Attributes which should stay: `path`, `shape`, `downscale_factor`, `array`. We don't need bit depth, metadata, timestamp...
- No dark subtraction, no interpolation of bad pixels, no `_validate_similarity`, no downscaled copy.
- No read tile.
- We do want to plot, however.
- Most of the heavy lifting will be in the `from_frame` method. It will do the following:
  
  - Check the `Frame`'s `dtype`.
  - Turn the `Frame` array into a grayscale of the same dtype.
  - Check the min/max value of the dtype (`np.iinfo(dtype).min/max`).
  - Turn the array to `float16`.
  - Rescale the array between 0 (dtype's min) and 1 (dtype's max).
  - Build a `ProxyMask` from the `Mask` with the same downscale factor.
  - Build a background array from the sky region with `photutils.background.Background2D(array, box_size, filter_size, mask=ProxyMask.array)`.
  - Subtract the background (converted to float16) from the array and subsequently set the masked region (foreground) to 0.0
  - That should be it - sky is completely free of background, the noise is centered around 0.0, stars are below 1.0, foreground is 0.0.
  - The box size (defaults to 16) and filter size (default to 1) are optional parameters to the class method, supplied from the config.
  - The proxy mask is discarded - only serves for downscaling the mask array.

## The rest of the repo

- The config and the tests are restructured for the new sub-packages/modules tree.
- Averaging code only works on `Frame` instances, producing new detached `Frame` instances, does not know anything about downscaling.
- The `StarDetector` only acts on the proxy instances, does not know anything about `Frame` and `Mask` instances.
  - The Star Detector's photometry helper module is redundant. There is no flux and local flux. Without the background, every star's flux is local.
  - All the defaults and configs are in the proxy units and pixels, nothing is in full-res. The StarDetector does not give any shit about how the full-res looked like.
- Check any other possible inconsistencies and problems that I have not seen.
