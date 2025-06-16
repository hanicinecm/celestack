"""A module defining the functionality for representing a segment of a frame."""

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go

import celestack._utils as utils


@dataclass(frozen=True)
class SegmentBox:
    """A class representing the rectangle, which will define the segment of a frame.

    The box is defined by its top-left coordinate (x1, y1) and bottom-right coordinate
    (x2, y2). All are integers - indices of the pixel in the image.
    The (x1, y1) coordinates are inclusive, while (x2, y2) coordinates are exclusive.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def x(self) -> int:
        """Get the x-coordinate of box center."""
        return (self.x1 + self.x2) // 2

    @property
    def y(self) -> int:
        """Get the y-coordinate of box center."""
        return (self.y1 + self.y2) // 2

    def to_polygon(self) -> np.ndarray:
        """Convert the box to a NumPy array representing its polygon.

        Returns:
            A NumPy array with the shape (5, 2) containing the coordinates of the box
            as a polygon.
        """
        return np.array(
            [
                [self.x1, self.y1],
                [self.x2, self.y1],
                [self.x2, self.y2],
                [self.x1, self.y2],
                [self.x1, self.y1],  # Closing the polygon
            ],
            dtype=np.int32,
        )


class Segment:
    """A class representing a segment of a frame.

    The segment is defined by its box (x1, y1, x2, y2) and the pixel array.
    The pixel array is a 2D NumPy array representing the pixel values of the segment.

    The segment class defines all sorts of methods which are focused on recognition of
    stars in the segment. The Segment should therefore only be instantiated with the
    part of the image with night sky and stars, as opposed to the foreground or other
    objects, such as dark frames, etc.
    """

    def __init__(
        self,
        box: SegmentBox,
        array: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> None:
        """Initialize the Segment object with a box and an array.

        The array should be a 2D NumPy array in grayscale 8-bit format (uint8).
        The segments are normally instantiated only from the sections of the
        `array_gs` (grayscale array) of the `Frame` objects.

        Args:
            box: The SegmentBox object defining the segment.
            array: The pixel array representing the segment - must be 2D uint8 and
                must have the shape consistent with the box.
            mask: Optional mask array for the segment. Defaults to None. If provided,
                it must have the same shape as the `array` and should be a boolean
                type, with True values indicating the foreground pixels to be masked
                out.
        """
        self.box = box
        self.array = array
        self.mask = mask

    def find_stars(
        self,
        density: float,
        fwhm: float,
        max_roundness: float,
        min_separation: float,
        init_thresh: float = 4.0,
    ) -> utils.StarsList:
        """Find stars in the segment and return them packaged in the StarsList.

        The stars are found using the DAOStarFinder algorithm from the photutils
        library, and the method intelligently adjusts the algorithm's threshold to find
        the specified final stars density.
        The threshold will be optimized to find double the number of stars requested
        and the stars will then be capped to the `n` brightest ones.

        The final threshold which lead to the number of stars needed for the target N
        is recorded in the returned dataframe.
        It can be used as the `init_thresh` parameter for the next call of
        the `find_stars` in the neighboring segments (either spatially or in the stack).

        All stars that are too close to other stars or to to an edge, are rejected.

        The stars are ordered in the stars list by their flux (brightest first).

        Args:
            density: The final density of the found stars in the segment, in
                stars per 10,000 sky pixels.
            fwhm: The full width at half maximum (FWHM) of the stars in [px].
                This is one of the parameters of the star finding algorithm and might
                have to be adjusted for each project.
            max_roundness: The maximum roundness of the stars. This is fed to the
                DAOStarFinder algorithm - all the stars with |roundness| > max_roundness
                will be rejected by DAOStarFinder.
            min_separation: The minimum separation between stars, in the units of fwhm.
                This is fed to the DAOStarFinder algorithm - all the stars with
                separation < min_separation * fwhm will be rejected by DAOStarFinder.
            init_thresh: The starting threshold for the star finding algorithm, in the
                number of standard deviations of the typical bacground noise.
                This is treated only as an initial guess, and the algorithm will adjust
                this to get to the target `n`.
                Optional, if not passed, a sensible default is provided.

        Returns:
            Stars detected in the segment, packaged in a StarsList object, ordered by
            their flux, brightest first.
            The positions of the stars are in the image coordinate system, i.e. the
            (x, y) coordinates are relative to the top-left corner of the parent image,
            rather than the segment box.

        TODO: Optimize with a binary search for the threshold.
        """
        # Calculate the number of stars to find:
        n_pixels = self.array.size
        if self.mask is not None:
            n_pixels -= np.sum(self.mask)
        n = round(n_pixels * density / 10000)

        # Find the sources:
        # Start with the `init_thresh * bkg_mad` threshold and iterate the threshold
        # until we find close to required number of stars:
        _sources = utils.StarsList.empty()
        delta_n = None
        thresh = init_thresh
        thresh_incr = 0.2
        target_n = 2 * n  # target number of stars (will be capped later)
        n_iters_stagnating = 0

        while True:
            # find the stars:
            new_sources = utils.find_stars(
                array=self.array,
                threshold=thresh,
                fwhm=fwhm,
                max_roundness=max_roundness,
                min_separation=min_separation,
                mask=self.mask,
            )
            # how far are we from the target number of sources?
            new_delta_n = len(new_sources) - target_n
            if delta_n is not None and abs(new_delta_n) > abs(delta_n):
                # we're further away from target than in the last iteration - stop...
                break
            if delta_n is not None and abs(new_delta_n) == abs(delta_n):
                # we're not getting closer to the target, but we don't want to stop
                # right away...
                n_iters_stagnating += 1
                if n_iters_stagnating > 3:
                    # ... so we stop after 3 iterations of stagnation
                    break
            else:
                n_iters_stagnating = 0

            delta_n = new_delta_n
            _sources = new_sources
            # adjust the threshold for the next iteration:
            thresh += thresh_incr if new_delta_n > 0 else -thresh_incr

        stars_list = _sources

        # If after all this we have no stars, just return the empty StarsList:
        if not stars_list:
            return stars_list

        # Sort by flux and cap the number of stars to `n`:
        stars_list.sort()
        stars_list.cap(n)

        # Absolutize the positions to the image coordinate system and return:
        stars_list.x += self.box.x1
        stars_list.y += self.box.y1

        return stars_list

    def plot(self) -> go.Figure:
        """Plot the segment.

        Returns:
            A Plotly figure object containing the image.
        """
        return utils.plot_image(self.array, interactive=True)
