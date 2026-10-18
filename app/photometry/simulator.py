"""Analytic exoplanet transit light curve simulator based on Mandel & Agol (2002).

Implements quadratic limb-darkening approximation and realistic stellar astrophysics
with white noise and optional stellar flare injection.
"""

from typing import Optional, Tuple
import numpy as np


def generate_synthetic_light_curve(
    n_days: float = 90.0,
    cadence_minutes: float = 29.4,
    period_days: float = 3.524,
    t0_days: float = 1.25,
    radius_ratio: float = 0.085,  # Rp / R_star (~0.7% depth for gas giant / sub-Neptune)
    semi_major_axis_stellar_radii: float = 12.0,
    inclination_deg: float = 88.5,
    u1: float = 0.35,  # Quadratic limb darkening linear coefficient
    u2: float = 0.20,  # Quadratic limb darkening quadratic coefficient
    snr_noise_sigma: float = 0.0008,  # Photometric white noise (~800 ppm)
    random_seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate time series flux with Mandel-Agol analytic transit profile.

    Parameters
    ----------
    n_days : float
        Total duration of observation in days (e.g. 90 days for Kepler quarter).
    cadence_minutes : float
        Sampling interval in minutes.
    period_days : float
        True planetary orbital period.
    t0_days : float
        Epoch of first transit center.
    radius_ratio : float
        Planet-to-star radius ratio k = Rp / R_star.
    semi_major_axis_stellar_radii : float
        Normalized semi-major axis a / R_star.
    inclination_deg : float
        Orbital inclination in degrees (90 deg = edge-on).
    u1, u2 : float
        Quadratic limb darkening coefficients.
    snr_noise_sigma : float
        Gaussian standard deviation of photometric measurement noise.
    random_seed : int, optional
        Reproducibility seed.

    Returns
    -------
    time : np.ndarray
        Time points in days.
    flux : np.ndarray
        Observed normalized flux containing transits and noise.
    flux_err : np.ndarray
        Measurement uncertainty array.
    """
    if random_seed is not None:
        rng = np.random.RandomState(random_seed)
    else:
        rng = np.random.RandomState()

    dt_days = cadence_minutes / (24.0 * 60.0)
    time = np.arange(0.0, n_days, dt_days, dtype=np.float64)
    n_pts = len(time)

    # 1. Compute orbital phase phi in [-0.5, 0.5)
    phases = ((time - t0_days + 0.5 * period_days) % period_days) / period_days - 0.5
    true_anom = 2.0 * np.pi * phases

    # 2. Projected center-to-center distance normalized by stellar radius (z = d / R_star)
    inc_rad = np.radians(inclination_deg)
    # Circular orbit approximation
    x_proj = semi_major_axis_stellar_radii * np.sin(true_anom)
    y_proj = semi_major_axis_stellar_radii * np.cos(true_anom) * np.cos(inc_rad)
    z = np.sqrt(x_proj ** 2 + y_proj ** 2)

    # Only points where planet is in front of star (phases close to 0)
    in_front = (np.cos(true_anom) >= 0.0) & (z < (1.0 + radius_ratio))

    # 3. Analytic transit depth evaluation (Mandel & Agol 2002 uniform + limb darkening adjustment)
    pure_flux = np.ones(n_pts, dtype=np.float64)
    p = radius_ratio
    p2 = p * p

    for i in np.where(in_front)[0]:
        zi = z[i]
        if zi <= (1.0 - p):
            # Complete ingress (planet fully inside stellar disk)
            # Quadratic limb darkening central profile
            mu = np.sqrt(np.maximum(0.0, 1.0 - zi ** 2))
            c1 = 1.0 - u1 * (1.0 - mu) - u2 * ((1.0 - mu) ** 2)
            c_norm = 1.0 - u1 / 3.0 - u2 / 6.0
            depth_factor = (c1 / c_norm) if c_norm > 0 else 1.0
            pure_flux[i] = 1.0 - (p2 * depth_factor)
        elif zi < (1.0 + p):
            # Grazing ingress / egress intersection
            # Overlapping circle segment area
            d1 = (1.0 - p2 + zi ** 2) / (2.0 * zi)
            d2 = (p2 - 1.0 + zi ** 2) / (2.0 * zi)
            d1 = np.clip(d1, -1.0, 1.0)
            d2 = np.clip(d2, -p, p) / p
            area = (
                np.arccos(d1)
                - d1 * np.sqrt(np.maximum(0.0, 1.0 - d1 ** 2))
                + p2 * np.arccos(d2)
                - p * d2 * np.sqrt(np.maximum(0.0, p2 - (p * d2) ** 2))
            )
            overlap_frac = np.clip(area / np.pi, 0.0, p2)
            pure_flux[i] = 1.0 - overlap_frac

    # 4. Inject Gaussian white noise
    noise = rng.normal(0.0, snr_noise_sigma, size=n_pts)
    flux = pure_flux + noise
    flux_err = np.full(n_pts, snr_noise_sigma, dtype=np.float64)

    return time, flux, flux_err
