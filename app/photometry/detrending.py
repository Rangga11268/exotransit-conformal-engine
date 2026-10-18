"""Robust light curve detrending to remove stellar variability without distorting transits."""

import numpy as np
from scipy.ndimage import median_filter


def robust_detrend_flux(
    time: np.ndarray,
    flux: np.ndarray,
    window_days: float = 1.5,
) -> np.ndarray:
    """Eliminate low-frequency stellar rotational variability and instrumental trends.

    Applies iterative running median filtering with outlier rejection,
    preserving short-duration transit depressions.

    Parameters
    ----------
    time : np.ndarray
        Observation timestamps in days.
    flux : np.ndarray
        Raw normalized photometric flux.
    window_days : float
        Filter kernel width in days.

    Returns
    -------
    np.ndarray
        Detrended, normalized flux with unit baseline (out-of-transit ~ 1.0).
    """
    dt = np.nanmedian(np.diff(time))
    if dt <= 0:
        return flux.copy()

    kernel_size = max(5, int(round(window_days / dt)))
    if kernel_size % 2 == 0:
        kernel_size += 1

    # First pass: median baseline
    trend = median_filter(flux, size=kernel_size, mode="reflect")

    # Second pass: mask deep negative outliers (transit candidate points)
    residuals = flux - trend
    std_est = 1.4826 * np.nanmedian(np.abs(residuals - np.nanmedian(residuals)))
    non_transit = residuals > (-2.0 * std_est)

    clean_flux = flux.copy()
    clean_flux[~non_transit] = trend[~non_transit]

    refined_trend = median_filter(clean_flux, size=kernel_size, mode="reflect")
    detrended = flux / np.maximum(1e-6, refined_trend)

    return detrended.astype(np.float64)
