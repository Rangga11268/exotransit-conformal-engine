"""Photometry package exports."""

from app.photometry.bls import BLSResult, BoxLeastSquaresEngine
from app.photometry.detrending import robust_detrend_flux
from app.photometry.phase_fold import (
    DualViewProjection,
    compute_dual_view_projection,
    generate_dual_views,
)
from app.photometry.simulator import (
    generate_synthetic_light_curve,
    simulate_transit_light_curve,
    inject_stellar_noise,
)

__all__ = [
    "BLSResult",
    "BoxLeastSquaresEngine",
    "robust_detrend_flux",
    "DualViewProjection",
    "compute_dual_view_projection",
    "generate_dual_views",
    "generate_synthetic_light_curve",
    "simulate_transit_light_curve",
    "inject_stellar_noise",
]
