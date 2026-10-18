"""Dual-view phase folding representation (Shallue & Vanderburg, 2018).

Constructs:
1. Global View: 2,001 uniform phase bins spanning full orbit [-0.5, +0.5].
2. Local View: 201 uniform bins zoomed into transit window [-2 * q, +2 * q].
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from app.config import DEFAULT_PHOTOMETRY, PhotometryConfig


@dataclass(frozen=True)
class DualViewProjection:
    """Fixed-dimension binned arrays for machine learning ingestion."""
    global_view: np.ndarray  # shape (2001,)
    local_view: np.ndarray  # shape (201,)
    global_phases: np.ndarray  # shape (2001,)
    local_phases: np.ndarray  # shape (201,)


def compute_dual_view_projection(
    time: np.ndarray,
    flux: np.ndarray,
    period_days: Optional[float] = None,
    t0_days: Optional[float] = None,
    duration_days: Optional[float] = None,
    config: PhotometryConfig = DEFAULT_PHOTOMETRY,
    period: Optional[float] = None,
    epoch: Optional[float] = None,
    duration: Optional[float] = None,
) -> DualViewProjection:
    period_days = period_days if period_days is not None else period
    t0_days = t0_days if t0_days is not None else epoch
    duration_days = duration_days if duration_days is not None else duration

    if period_days is None or t0_days is None or duration_days is None:
        raise ValueError("Must provide period_days, t0_days, and duration_days.")


    """Project phase-folded light curve into Global View and Local View representations.

    Parameters
    ----------
    time : np.ndarray
        Timestamps in days.
    flux : np.ndarray
        Normalized flux.
    period_days : float
        Orbital period in days.
    t0_days : float
        Transit center epoch in days.
    duration_days : float
        Estimated transit duration in days.
    config : PhotometryConfig
        Bin counts and zoom configuration.

    Returns
    -------
    DualViewProjection
        Global View (2001 bins) and Local View (201 bins) normalized arrays.
    """
    P = float(period_days)
    t0 = float(t0_days)
    q_frac = float(duration_days) / P

    # 1. Compute folded phase in [-0.5, 0.5) centered at transit t0
    phases = ((time - t0 + 0.5 * P) % P) / P - 0.5

    # -------------------------------------------------------------
    # A. Global View: 2001 bins across [-0.5, +0.5]
    # -------------------------------------------------------------
    n_global = config.global_bins
    global_edges = np.linspace(-0.5, 0.5, n_global + 1)
    global_centers = 0.5 * (global_edges[:-1] + global_edges[1:])

    bin_indices_g = np.digitize(phases, global_edges) - 1
    valid_g = (bin_indices_g >= 0) & (bin_indices_g < n_global)

    counts_g = np.bincount(bin_indices_g[valid_g], minlength=n_global)
    sum_flux_g = np.bincount(bin_indices_g[valid_g], weights=flux[valid_g], minlength=n_global)

    global_view = np.ones(n_global, dtype=np.float32)
    has_points_g = counts_g > 0
    global_view[has_points_g] = (sum_flux_g[has_points_g] / counts_g[has_points_g]).astype(np.float32)

    # Interpolate empty bins if any
    if np.any(~has_points_g) and np.any(has_points_g):
        global_view[~has_points_g] = np.interp(
            global_centers[~has_points_g],
            global_centers[has_points_g],
            global_view[has_points_g],
        )

    # -------------------------------------------------------------
    # B. Local View: 201 bins zoomed into [-2 * q, +2 * q]
    # -------------------------------------------------------------
    n_local = config.local_bins
    half_width = max(0.01, config.local_window_durations * q_frac)
    local_edges = np.linspace(-half_width, half_width, n_local + 1)
    local_centers = 0.5 * (local_edges[:-1] + local_edges[1:])

    in_local = (phases >= -half_width) & (phases <= half_width)
    bin_indices_l = np.digitize(phases[in_local], local_edges) - 1
    valid_l = (bin_indices_l >= 0) & (bin_indices_l < n_local)

    counts_l = np.bincount(bin_indices_l[valid_l], minlength=n_local)
    sum_flux_l = np.bincount(bin_indices_l[valid_l], weights=flux[in_local][valid_l], minlength=n_local)

    local_view = np.ones(n_local, dtype=np.float32)
    has_points_l = counts_l > 0
    local_view[has_points_l] = (sum_flux_l[has_points_l] / counts_l[has_points_l]).astype(np.float32)

    # Interpolate empty bins if any
    if np.any(~has_points_l) and np.any(has_points_l):
        local_view[~has_points_l] = np.interp(
            local_centers[~has_points_l],
            local_centers[has_points_l],
            local_view[has_points_l],
        )

    return DualViewProjection(
        global_view=global_view,
        local_view=local_view,
        global_phases=global_centers.astype(np.float32),
        local_phases=local_centers.astype(np.float32),
    )


generate_dual_views = compute_dual_view_projection

