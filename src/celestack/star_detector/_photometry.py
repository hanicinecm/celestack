"""Local-background aperture photometry.

Corrects raw DAOStarFinder ``flux`` for spatially varying sky brightness
(e.g. light-pollution gradients toward the horizon) by subtracting a
local sky estimate measured in an annulus around each detection.
"""

from __future__ import annotations

import numpy as np
from photutils.aperture import (
    ApertureStats,
    CircularAnnulus,
    CircularAperture,
    aperture_photometry,
)

_APERTURE_FWHM_FACTOR = 1.5  # aperture radius as a multiple of FWHM
_ANNULUS_INNER_FWHM_FACTOR = 2.5  # sky annulus inner radius
_ANNULUS_OUTER_FWHM_FACTOR = 4.0  # sky annulus outer radius


def compute_local_flux(
    image: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    fwhm: float,
) -> np.ndarray:
    """Compute background-subtracted aperture flux for each star.

    For each ``(x, y)`` position, sums pixels in a circular aperture of
    radius ``1.5 * fwhm`` and subtracts the per-star sky estimate (median
    of the pixels in an annulus between ``2.5 * fwhm`` and ``4.0 * fwhm``)
    scaled by the aperture area.

    Args:
        image: 2D grayscale array.
        xs: X centroids in image-pixel coordinates.
        ys: Y centroids in image-pixel coordinates.
        fwhm: FWHM in image pixels.

    Returns:
        Float32 array of length ``len(xs)`` with background-subtracted
        flux per star.  Positions whose annulus falls fully off-image are
        returned as ``0.0``.
    """
    if len(xs) == 0:
        return np.array([], dtype=np.float32)

    positions = np.column_stack([xs, ys])
    r_ap = _APERTURE_FWHM_FACTOR * fwhm
    r_in = _ANNULUS_INNER_FWHM_FACTOR * fwhm
    r_out = _ANNULUS_OUTER_FWHM_FACTOR * fwhm

    aperture = CircularAperture(positions, r=r_ap)
    annulus = CircularAnnulus(positions, r_in=r_in, r_out=r_out)

    phot = aperture_photometry(image, aperture)
    bg_median = ApertureStats(image, annulus).median

    aperture_sum = np.asarray(phot["aperture_sum"], dtype=np.float64)
    bg_median = np.asarray(bg_median, dtype=np.float64)
    flux_local = aperture_sum - bg_median * aperture.area
    flux_local = np.nan_to_num(flux_local, nan=0.0, posinf=0.0, neginf=0.0)
    return flux_local.astype(np.float32)
