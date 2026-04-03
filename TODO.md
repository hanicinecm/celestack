# TODO

- [ ] **Optimize `average_frames`**: explore multithreading for tile I/O and
  multiprocessing for numeric aggregation. Known issue: averaging 100 frames
  takes much more than 10× the time of averaging 10 frames — the scaling is
  super-linear when it should be roughly linear.
