"""ExoTransit-Conformal Engine configuration dataclasses."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PhotometryConfig:
    """Astrophysical photometry search and grid parameters."""
    min_period_days: float = 0.5
    max_period_days: float = 30.0
    n_periods: int = 1000
    duration_fractions: Tuple[float, ...] = (0.01, 0.02, 0.04, 0.08)
    global_bins: int = 2001
    local_bins: int = 201
    local_window_durations: float = 2.0  # window is [-local_window_durations * q, +local_window_durations * q]
    detrend_window_days: float = 1.5
    sample_rate_cadence_minutes: float = 29.4  # Standard Kepler long cadence (~30 min)


@dataclass(frozen=True)
class ConformalConfig:
    """Distribution-free split conformal prediction configuration."""
    nominal_error_rate: float = 0.05  # alpha = 0.05 implies 95% coverage guarantee
    n_calibration_samples: int = 1000
    ambiguity_threshold: float = 0.50


DEFAULT_PHOTOMETRY = PhotometryConfig()
DEFAULT_CONFORMAL = ConformalConfig()
