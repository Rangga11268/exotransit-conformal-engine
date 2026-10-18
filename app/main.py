"""FastAPI Production REST API for ExoTransit-Conformal Engine."""

import time
import os
from contextlib import asynccontextmanager
from typing import Optional
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


from app.schemas.transit import (
    TransitDetectionRequest,
    TransitDetectionResponse,
    BLSSummary,
    CalibrateRequest,
    CalibrateResponse,
    HealthResponse,
)
from app.photometry.bls import BoxLeastSquaresEngine, BLSResult
from app.photometry.detrending import robust_detrend_flux
from app.photometry.phase_fold import compute_dual_view_projection
from app.models.classifier import DualBranchClassifier, generate_synthetic_dataset
from app.conformal.calibrator import SplitConformalCalibrator


# Global singletons
classifier: Optional[DualBranchClassifier] = None
calibrator: Optional[SplitConformalCalibrator] = None
bls_engine: Optional[BoxLeastSquaresEngine] = None

CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "conformal_calibration.json")


def _bootstrap_calibrator(alpha: float = 0.05, n_samples: int = 250) -> SplitConformalCalibrator:
    """Helper to ensure a calibrated conformal layer is always ready."""
    cal = SplitConformalCalibrator(nominal_error_rate=alpha)
    if os.path.exists(CALIBRATION_FILE):
        try:
            return SplitConformalCalibrator.load(CALIBRATION_FILE)
        except Exception:
            pass

    # Calibrate on synthetic representative samples
    global classifier
    if classifier is None:
        classifier = DualBranchClassifier()

    X, y = generate_synthetic_dataset(n_samples=n_samples, random_seed=999)
    probs = classifier.predict_proba_from_features(X)
    cal.calibrate(probs, y)
    try:
        cal.save(CALIBRATION_FILE)
    except Exception:
        pass
    return cal


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and teardown handler."""
    global classifier, calibrator, bls_engine
    classifier = DualBranchClassifier()
    bls_engine = BoxLeastSquaresEngine()
    calibrator = _bootstrap_calibrator()
    yield


app = FastAPI(
    title="ExoTransit-Conformal Engine",
    description="Vectorized Box Least Squares & Distribution-Free Split Conformal Prediction for Exoplanet Discovery",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


INDEX_HTML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html")


@app.get("/", tags=["General"])
async def root():
    """Engine root metadata."""
    return {
        "engine": "ExoTransit-Conformal",
        "description": "Finite-sample error-guaranteed exoplanet transit detection",
        "status": "operational",
        "gui_url": "/gui",
        "docs_url": "/docs",
        "version": "1.0.0"
    }


@app.get("/gui", tags=["General"])
async def get_gui():
    """Serves pure HTML/CSS/JS observatory dashboard."""
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML, media_type="text/html")
    raise HTTPException(status_code=404, detail="index.html not found")



@app.get("/v1/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """System health check and conformal quantile status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=(classifier is not None and classifier.booster is not None),
        is_calibrated=(calibrator is not None and calibrator.q_hat is not None),
        active_q_hat=calibrator.q_hat if calibrator else None,
        nominal_coverage=f"{(calibrator.target_coverage * 100):.1f}%" if calibrator else "N/A"
    )


@app.post("/v1/transit/detect", response_model=TransitDetectionResponse, tags=["Inference"])
async def detect_transit(req: TransitDetectionRequest):
    """End-to-end exoplanet transit discovery pipeline.
    
    Performs:
    1. Input validation & sanity check
    2. Robust detrending (optional)
    3. SIMD Box Least Squares (BLS) periodogram search
    4. Dual-view phase-folding (Global & Local views)
    5. Dual-branch morphological classification
    6. Distribution-free split conformal prediction set routing
    """
    t_start = time.perf_counter()

    time_arr = np.asarray(req.time, dtype=np.float64)
    flux_arr = np.asarray(req.flux, dtype=np.float64)

    if len(time_arr) != len(flux_arr):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Array length mismatch: time ({len(time_arr)}) vs flux ({len(flux_arr)})"
        )
    if len(time_arr) < 50:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Time-series too short for BLS periodicity: got {len(time_arr)} points, minimum is 50"
        )
    if not np.all(np.isfinite(time_arr)) or not np.all(np.isfinite(flux_arr)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input data contains non-finite values (NaN or Inf)"
        )

    # 1. Detrending
    if req.detrend:
        flux_clean = robust_detrend_flux(time_arr, flux_arr)
    else:
        flux_clean = flux_arr

    # 2. Vectorized BLS
    assert bls_engine is not None
    bls_res = bls_engine.search(time_arr, flux_clean)

    # 3. Dual-View Phase Folding
    views = compute_dual_view_projection(
        time=time_arr,
        flux=flux_clean,
        period_days=bls_res.best_period,
        t0_days=bls_res.best_epoch,
        duration_days=bls_res.best_duration
    )

    # 4. Classification
    assert classifier is not None
    prob_vec = classifier.predict_proba(
        global_view=views.global_view,
        local_view=views.local_view,
        bls_result=bls_res
    )
    p_fp = float(prob_vec[0])
    p_planet = float(prob_vec[1])

    # 5. Conformal Prediction Set
    assert calibrator is not None
    # If client requested custom alpha different from active calibrator
    if abs(req.alpha - calibrator.alpha) > 1e-4:
        active_cal = SplitConformalCalibrator(nominal_error_rate=req.alpha)
        # Use existing calibration scores if available
        if calibrator.calibration_scores is not None:
            active_cal.calibration_scores = calibrator.calibration_scores
            active_cal.n_calibration = calibrator.n_calibration
            level = np.ceil((active_cal.n_calibration + 1) * (1.0 - req.alpha)) / active_cal.n_calibration
            level = min(1.0, max(0.0, level))
            active_cal.q_hat = float(np.quantile(active_cal.calibration_scores, level, method="higher"))
        else:
            active_cal = calibrator
    else:
        active_cal = calibrator

    conformal_sets = active_cal.predict_set(
        probabilities=np.array([[p_fp, p_planet]]),
        candidate_ids=[req.candidate_id or "KOI-001"],
        snrs=[float(bls_res.sde)]
    )
    conf_res = conformal_sets[0].to_dict()

    # Radius ratio Rp / R_star ~ sqrt(transit_depth)
    radius_ratio_est = float(np.sqrt(max(0.0, bls_res.best_depth)))
    total_time_ms = (time.perf_counter() - t_start) * 1000.0

    return TransitDetectionResponse(
        candidate_id=req.candidate_id or "KOI-001",
        bls_summary=BLSSummary(
            period_days=round(float(bls_res.best_period), 4),
            t0_days=round(float(bls_res.best_epoch), 4),
            duration_days=round(float(bls_res.best_duration), 4),
            depth=round(float(bls_res.best_depth), 6),
            sde=round(float(bls_res.sde), 2),
            snr=round(float(bls_res.sde), 2),
        ),
        estimated_radius_ratio=round(radius_ratio_est, 4),
        probabilities={
            "FALSE_POSITIVE": round(p_fp, 4),
            "PLANET": round(p_planet, 4),
        },
        conformal_prediction=conf_res,
        processing_time_ms=round(total_time_ms, 2),
    )


@app.post("/v1/conformal/calibrate", response_model=CalibrateResponse, tags=["Administration"])
async def calibrate_engine(req: CalibrateRequest):
    """Recalibrate split conformal quantile for custom error rates."""
    global calibrator, classifier
    if classifier is None:
        classifier = DualBranchClassifier()

    cal = SplitConformalCalibrator(nominal_error_rate=req.alpha)
    X, y = generate_synthetic_dataset(n_samples=req.n_synthetic_samples, random_seed=int(req.alpha * 10000))
    probs = classifier.predict_proba_from_features(X)
    q_hat = cal.calibrate(probs, y)
    calibrator = cal
    try:
        cal.save(CALIBRATION_FILE)
    except Exception:
        pass

    return CalibrateResponse(
        status="recalibrated_successfully",
        quantile_q_hat=round(q_hat, 4),
        nominal_error_rate=req.alpha,
        target_coverage=round(1.0 - req.alpha, 4),
        n_calibration=len(y)
    )


@app.get("/v1/nasa/benchmarks", tags=["NASA Empirical Data"])
async def get_nasa_benchmarks():
    """Retrieve curated NASA Kepler archive benchmark targets for empirical validation."""
    from app.data.nasa_samples import NASA_KEPLER_BENCHMARKS
    return {
        "status": "success",
        "total_targets": len(NASA_KEPLER_BENCHMARKS),
        "source": "NASA Exoplanet Archive (IPAC/Caltech) Cumulative Table",
        "benchmarks": NASA_KEPLER_BENCHMARKS
    }

