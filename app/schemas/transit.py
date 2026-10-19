"""Pydantic v2 request/response schemas for ExoTransit-Conformal Engine."""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


class TransitDetectionRequest(BaseModel):
    """Input payload containing photometric time-series."""
    time: List[float] = Field(..., description="Timestamps in days (e.g. BJD or BKJD)")
    flux: List[float] = Field(..., description="Normalized relative flux measurements")
    candidate_id: Optional[str] = Field("KOI-USER-001", description="Astronomical candidate ID")
    alpha: float = Field(0.05, ge=0.001, le=0.50, description="Nominal error rate tolerance (default 0.05 for 95% coverage)")
    detrend: Optional[bool] = Field(True, description="Whether to apply running median detrending before BLS")


class BLSSummary(BaseModel):
    """Photometric Box Least Squares periodicity findings."""
    period_days: float
    t0_days: float
    duration_days: float
    depth: float
    sde: float
    snr: float


class TransitDetectionResponse(BaseModel):
    """Complete exoplanet detection, classification, and conformal set response."""
    candidate_id: str
    bls_summary: BLSSummary
    estimated_radius_ratio: float
    probabilities: Dict[str, float]
    conformal_prediction: Dict[str, Any]
    processing_time_ms: float


class CalibrateRequest(BaseModel):
    """Administrative recalibration request payload."""
    alpha: float = Field(0.05, ge=0.001, le=0.50, description="Target error rate (e.g. 0.05 for 95% coverage)")
    n_synthetic_samples: int = Field(300, ge=50, le=2000, description="Synthetic calibration set size if no external dataset provided")


class CalibrateResponse(BaseModel):
    """Recalibration acknowledgment response."""
    status: str
    quantile_q_hat: float
    nominal_error_rate: float
    target_coverage: float
    n_calibration: int


class HealthResponse(BaseModel):
    """System health and diagnostic status."""
    model_config = {"protected_namespaces": ()}
    status: str
    version: str
    model_loaded: bool
    is_calibrated: bool
    active_q_hat: Optional[float]
    nominal_coverage: str
    # Calibration provenance (Priority 3)
    calibration_alpha: Optional[float] = None
    calibration_n_samples: Optional[int] = None
    calibration_source: Optional[str] = None
    model_version: Optional[str] = None

