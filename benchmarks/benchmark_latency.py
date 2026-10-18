"""Latency Benchmark for ExoTransit-Conformal Engine.

Measures P50, P90, and P99 latency percentiles across the full inference pipeline:
Detrending -> SIMD BLS -> Dual-View Folding -> Dual-Branch Classifier -> Conformal Calibration.
"""

import time
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import numpy as np
from app.photometry.simulator import simulate_transit_light_curve, inject_stellar_noise
from app.photometry.detrending import robust_detrend_flux
from app.photometry.bls import BoxLeastSquaresEngine
from app.photometry.phase_fold import compute_dual_view_projection
from app.models.classifier import DualBranchClassifier
from app.conformal.calibrator import SplitConformalCalibrator


def run_latency_benchmark(n_trials: int = 20) -> dict:
    print("=" * 70, flush=True)
    print(">>> EXOTRANSIT-CONFORMAL: END-TO-END CPU LATENCY BENCHMARK <<<", flush=True)
    print("=" * 70, flush=True)


    # Initialize engines
    bls_engine = BoxLeastSquaresEngine()
    classifier = DualBranchClassifier()
    calibrator = SplitConformalCalibrator(nominal_error_rate=0.05)
    calibrator.q_hat = 0.42 # calibrated default

    # Prepare standard 30-day Kepler cadence light curve (~1,470 photometric points)
    cadence_days = 29.4 / (24.0 * 60.0)
    time_arr = np.arange(0.0, 30.0, cadence_days)
    period = 3.524
    epoch = 0.8
    duration = 0.12
    raw_flux = simulate_transit_light_curve(
        time=time_arr,
        period=period,
        epoch=epoch,
        duration=duration,
        radius_ratio=0.05,
    )
    flux_arr = inject_stellar_noise(raw_flux, noise_std=3e-4, random_seed=42)

    print(f"Photometric points per trial: {len(time_arr):,}")
    print(f"Benchmark trials: {n_trials}")
    print("Warming up JIT & memory caches...")

    # Warmup
    for _ in range(5):
        clean_flux = robust_detrend_flux(time_arr, flux_arr)
        bls_res = bls_engine.search(time_arr, clean_flux)
        views = compute_dual_view_projection(time_arr, clean_flux, bls_res.best_period, bls_res.best_t0, bls_res.best_duration)
        probs = classifier.predict_proba(views.global_view, views.local_view, bls_res)
        _ = calibrator.predict_set(np.atleast_2d(probs))

    # Benchmark iterations
    times_detrend = []
    times_bls = []
    times_folding = []
    times_classifier = []
    times_conformal = []
    times_total = []

    print("Benchmarking sequential iterations...")
    for _ in range(n_trials):
        t0 = time.perf_counter()
        
        t_sub = time.perf_counter()
        clean_flux = robust_detrend_flux(time_arr, flux_arr)
        times_detrend.append((time.perf_counter() - t_sub) * 1000.0)

        t_sub = time.perf_counter()
        bls_res = bls_engine.search(time_arr, clean_flux)
        times_bls.append((time.perf_counter() - t_sub) * 1000.0)

        t_sub = time.perf_counter()
        views = compute_dual_view_projection(time_arr, clean_flux, bls_res.best_period, bls_res.best_t0, bls_res.best_duration)
        times_folding.append((time.perf_counter() - t_sub) * 1000.0)

        t_sub = time.perf_counter()
        probs = classifier.predict_proba(views.global_view, views.local_view, bls_res)
        times_classifier.append((time.perf_counter() - t_sub) * 1000.0)

        t_sub = time.perf_counter()
        _ = calibrator.predict_set(np.atleast_2d(probs))
        times_conformal.append((time.perf_counter() - t_sub) * 1000.0)

        times_total.append((time.perf_counter() - t0) * 1000.0)

    p50 = float(np.percentile(times_total, 50))
    p90 = float(np.percentile(times_total, 90))
    p99 = float(np.percentile(times_total, 99))
    mean_lat = float(np.mean(times_total))
    throughput_pts_sec = (len(time_arr) * n_trials) / (np.sum(times_total) / 1000.0)

    ml_pipeline = np.array(times_detrend) + np.array(times_folding) + np.array(times_classifier) + np.array(times_conformal)
    ml_p50 = float(np.percentile(ml_pipeline, 50))
    ml_p90 = float(np.percentile(ml_pipeline, 90))
    ml_p99 = float(np.percentile(ml_pipeline, 99))

    print("\n" + "-" * 70)
    print("STAGE LATENCY BREAKDOWN (Mean):")
    print(f"  1. Robust Median Detrending: {np.mean(times_detrend):.2f} ms")
    print(f"  2. Vectorized SIMD BLS:       {np.mean(times_bls):.2f} ms")
    print(f"  3. Dual-View Phase Folding:  {np.mean(times_folding):.2f} ms")
    print(f"  4. Dual-Branch Classifier:   {np.mean(times_classifier):.2f} ms")
    print(f"  5. Conformal Prediction Set: {np.mean(times_conformal):.2f} ms")
    print("-" * 70)
    print("ML PIPELINE (Detrend + Fold + Model + Conformal - PRD AC-4):")
    print(f"  P50: {ml_p50:.2f} ms | P90: {ml_p90:.2f} ms | P99: {ml_p99:.2f} ms (SLA < 20.0 ms)")
    print("-" * 70)
    print("FULL END-TO-END PIPELINE (Including 1,000-period BLS scan):")
    print(f"  P50 (Median):   {p50:.2f} ms")
    print(f"  P90:            {p90:.2f} ms")
    print(f"  P99:            {p99:.2f} ms")
    print(f"  Mean:           {mean_lat:.2f} ms")
    print(f"  Throughput:     {throughput_pts_sec:,.0f} photometric pts/sec")
    print("-" * 70)

    passed_sla = ml_p99 < 20.0
    status_str = f"[PASSED] ML Pipeline P99 ({ml_p99:.2f} ms) < 20.0 ms SLA" if passed_sla else f"[FAILED] ({ml_p99:.2f} ms >= 20.0 ms)"
    print(f"SLA Verification: {status_str}")
    print("=" * 70)

    return {
        "ml_p99_ms": ml_p99,
        "full_p99_ms": p99,
        "mean_ms": mean_lat,
        "throughput_pts_sec": throughput_pts_sec,
        "passed_sla": passed_sla,
    }



if __name__ == "__main__":
    run_latency_benchmark()
