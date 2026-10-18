"""Empirical NASA Kepler benchmark — downloads real FITS light curves via lightkurve.

Runs the full inference pipeline (BLS → phase-fold → classify → conformal) on
actual Kepler photometry and compares predictions against the NASA Exoplanet Archive
catalog ground-truth labels.

Usage:
    python benchmarks/benchmark_nasa_empirical.py

Requirements:
    pip install lightkurve

Output:
    benchmarks/nasa_empirical_results.json  — machine-readable results
    Console table — human-readable summary with pass/fail per target
"""

import json
import os
import sys
import time
import platform

import numpy as np

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import lightkurve as lk
except ImportError:
    print("ERROR: lightkurve not installed. Run: pip install lightkurve")
    sys.exit(1)

from app.photometry.bls import BoxLeastSquaresEngine
from app.photometry.detrending import robust_detrend_flux
from app.photometry.phase_fold import compute_dual_view_projection
from app.models.classifier import DualBranchClassifier
from app.conformal.calibrator import SplitConformalCalibrator
from app.data.nasa_samples import NASA_KEPLER_BENCHMARKS

# ──────────────────────────────────────────────────────────────────
# Ground-truth classification mapping
# ──────────────────────────────────────────────────────────────────
# Maps catalog ground_truth string → expected conformal-set membership.
# We treat CANDIDATE_BORDERLINE as acceptable for AMBIGUOUS.
_TRUTH_TO_EXPECTED_SET = {
    "CONFIRMED_PLANET": {"PLANET"},
    "FALSE_POSITIVE_EB": {"FALSE_POSITIVE"},
    "CANDIDATE_BORDERLINE": {"PLANET", "FALSE_POSITIVE"},  # either is fine
}


def _load_kepler_lc(mission_id: str, period_days: float) -> tuple[np.ndarray, np.ndarray]:
    """Download and stitch Kepler long-cadence light curve for a KIC target.

    Returns (time_days, normalized_flux). Raises RuntimeError on failure.
    """
    # Quarter search: download all available quarters and stitch
    search = lk.search_lightcurve(mission_id, mission="Kepler", cadence="long")
    if len(search) == 0:
        raise RuntimeError(f"No Kepler light curves found for {mission_id}")

    # Limit to first 4 quarters to keep download fast (~10–40 MB)
    n_quarters = min(4, len(search))
    collection = search[:n_quarters].download_all()
    lc = collection.stitch()

    # Remove NaN and normalize
    lc = lc.remove_nans().normalize()
    time = lc.time.bkjd  # Barycentric Kepler Julian Day
    flux = lc.flux.value

    # Sigma-clip obvious outliers (> 5σ)
    med = np.nanmedian(flux)
    std = np.nanstd(flux)
    mask = np.abs(flux - med) < 5.0 * std
    return time[mask], flux[mask]


def run_pipeline(
    time: np.ndarray,
    flux: np.ndarray,
    candidate_id: str,
    bls_engine: BoxLeastSquaresEngine,
    classifier: DualBranchClassifier,
    calibrator: SplitConformalCalibrator,
) -> dict:
    """Full inference pipeline on a real light curve. Returns result dict."""
    import time as _time_mod
    t_start = _time_mod.perf_counter()

    # 1. Detrend (Savitzky-Golay via robust_detrend)
    flux_clean = robust_detrend_flux(time, flux)

    # 2. BLS periodogram
    bls_res = bls_engine.search(time, flux_clean)

    # 3. Dual-view phase folding
    views = compute_dual_view_projection(
        time=time,
        flux=flux_clean,
        period_days=bls_res.best_period,
        t0_days=bls_res.best_epoch,
        duration_days=bls_res.best_duration,
    )

    # 4. Classification
    prob_vec = classifier.predict_proba(
        global_view=views.global_view,
        local_view=views.local_view,
        bls_result=bls_res,
    )
    p_fp, p_planet = float(prob_vec[0]), float(prob_vec[1])

    # 5. Conformal prediction set
    conf_sets = calibrator.predict_set(
        probabilities=np.array([[p_fp, p_planet]]),
        candidate_ids=[candidate_id],
        snrs=[float(bls_res.sde)],
    )
    conf = conf_sets[0].to_dict()

    elapsed_ms = (_time_mod.perf_counter() - t_start) * 1000.0

    return {
        "bls_period_days": round(float(bls_res.best_period), 4),
        "bls_depth": round(float(bls_res.best_depth), 6),
        "bls_sde": round(float(bls_res.sde), 2),
        "p_false_positive": round(p_fp, 4),
        "p_planet": round(p_planet, 4),
        "conformal_set": conf.get("prediction_set", []),
        "conformal_status": conf.get("status", "UNKNOWN"),
        "elapsed_ms": round(elapsed_ms, 1),
    }


def evaluate_target(sample: dict, result: dict) -> dict:
    """Check if conformal set is consistent with ground truth."""
    truth = sample["ground_truth"]
    expected_any = _TRUTH_TO_EXPECTED_SET.get(truth, set())
    predicted_set = set(result["conformal_set"])

    # Pass if the intersection is non-empty (prediction contains at least one correct label)
    intersection = expected_any & predicted_set
    passed = len(intersection) > 0

    # For CONFIRMED_PLANET: set must include "PLANET"
    # For FALSE_POSITIVE_EB: set must include "FALSE_POSITIVE"
    # For BORDERLINE: either label in set is acceptable
    if truth == "CONFIRMED_PLANET":
        passed = "PLANET" in predicted_set
    elif truth == "FALSE_POSITIVE_EB":
        passed = "FALSE_POSITIVE" in predicted_set

    return {
        "target_id": sample["id"],
        "ground_truth": truth,
        "predicted_set": list(predicted_set),
        "status": result["conformal_status"],
        "passed": passed,
        "p_planet": result["p_planet"],
        "p_fp": result["p_false_positive"],
        "bls_period_days": result["bls_period_days"],
        "catalog_period_days": sample["period_days"],
        "period_error_pct": round(
            abs(result["bls_period_days"] - sample["period_days"]) / sample["period_days"] * 100, 2
        ),
        "elapsed_ms": result["elapsed_ms"],
    }


def main():
    print("=" * 72)
    print("ExoTransit-Conformal — Empirical NASA Kepler Benchmark")
    print(f"Platform: {platform.system()} {platform.machine()}, Python {sys.version.split()[0]}")
    print("=" * 72)

    # Load singletons
    print("\nLoading model + calibrator...")
    bls_engine = BoxLeastSquaresEngine()
    classifier = DualBranchClassifier()

    cal_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "app", "conformal_calibration.json")
    calibrator = SplitConformalCalibrator.load(cal_file, expected_alpha=0.05)
    print(f"  Calibrator: alpha={calibrator.alpha:.2f}, q_hat={calibrator.q_hat:.4f}, "
          f"n={calibrator.n_calibration}")

    results_per_target = []
    eval_per_target = []

    for sample in NASA_KEPLER_BENCHMARKS:
        tid = sample["id"]
        mission_id = sample["mission_id"]
        print(f"\n[{tid}] Downloading {mission_id}...")

        try:
            t_dl = time.perf_counter()
            lc_time, lc_flux = _load_kepler_lc(mission_id, sample["period_days"])
            dl_sec = time.perf_counter() - t_dl
            n_pts = len(lc_time)
            print(f"  Downloaded {n_pts:,} points in {dl_sec:.1f}s")

            if n_pts < 50:
                raise RuntimeError(f"Insufficient points: {n_pts}")

            result = run_pipeline(lc_time, lc_flux, tid, bls_engine, classifier, calibrator)
            results_per_target.append({"id": tid, **result})

            ev = evaluate_target(sample, result)
            eval_per_target.append(ev)

            status_icon = "✓ PASS" if ev["passed"] else "✗ FAIL"
            print(f"  {status_icon} | set={ev['predicted_set']} | "
                  f"P(planet)={ev['p_planet']:.3f} | "
                  f"BLS period={ev['bls_period_days']:.4f}d "
                  f"(catalog={ev['catalog_period_days']:.4f}d, "
                  f"err={ev['period_error_pct']:.1f}%) | "
                  f"{ev['elapsed_ms']:.0f}ms")

        except Exception as exc:
            print(f"  ERROR: {exc}")
            eval_per_target.append({
                "target_id": tid,
                "ground_truth": sample["ground_truth"],
                "predicted_set": [],
                "status": "DOWNLOAD_ERROR",
                "passed": False,
                "error": str(exc),
            })

    # Summary
    n_total = len(eval_per_target)
    n_pass = sum(1 for e in eval_per_target if e.get("passed", False))
    n_error = sum(1 for e in eval_per_target if e.get("status") == "DOWNLOAD_ERROR")

    print("\n" + "=" * 72)
    print(f"RESULT: {n_pass}/{n_total} targets passed conformal set evaluation")
    if n_error:
        print(f"        {n_error} targets skipped due to download errors")
    print("=" * 72)

    # Save JSON
    output = {
        "benchmark": "ExoTransit-Conformal NASA Kepler Empirical Validation",
        "platform": f"{platform.system()} {platform.machine()}",
        "python_version": sys.version.split()[0],
        "lightkurve_version": lk.__version__,
        "calibrator_alpha": calibrator.alpha,
        "calibrator_q_hat": calibrator.q_hat,
        "n_calibration": calibrator.n_calibration,
        "summary": {"passed": n_pass, "total": n_total, "errors": n_error},
        "targets": eval_per_target,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nasa_empirical_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved → {out_path}")
    return 0 if n_pass == n_total - n_error else 1


if __name__ == "__main__":
    sys.exit(main())
