# TODO

- [ ] **Get the `dtype`s under control**: the images are loaded into the arrays
  of the correct data types, typically `uint16`, or `uint8` for the proxy images.
  However, when doing numerics on them (`mean`, `median`, `std`, ...) these will
  be converted to `float64` which is very wasteful.
  The float depth should be systematically tied to the image bit-depth for any
  numerical operations on the arrays.
- [ ] **Optimize `average_frames`**: explore multithreading for tile I/O and
  multiprocessing for numeric aggregation. Known issue: averaging 100 frames
  takes much more than 10× the time of averaging 10 frames — the scaling is
  super-linear when it should be roughly linear.
