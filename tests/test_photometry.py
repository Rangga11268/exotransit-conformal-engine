"""Unit test suite for Milestone 1: Photometry, BLS, Detrending, and Phase-Folding."""

import pytest
import numpy as np

from app.config import DEFAULT_PHOTOMETRY
from app.photometry.bls import BoxLeastSquaresEngine
from app.photometry.detrending import robust_detrend_flux
from app.photometry.phase_fold import compute_dual_view_projection
from app.photometry.simulator import generate_synthetic_light_curve


class TestMilestone1Photometry:
    """Test suite covering astrophysical light curve simulation, detrending, and BLS."""

    def test_synthetic_mandel_agol_generator(self):
        """Verify synthetic light curve exhibits expected transit dip and duration."""
        time, flux, flux_err = generate_synthetic_light_curve(
            n_days=30.0,
            cadence_minutes=29.4,
            period_days=4.0,
            t0_days=1.0,
            radius_ratio=0.10,  # ~1% depth
            snr_noise_sigma=0.0005,
            random_seed=123,
        )

        assert len(time) == len(flux) == len(flux_err)
        assert len(time) > 1000

        # Out-of-transit flux baseline should be close to 1.0
        median_flux = np.median(flux)
        assert np.isclose(median_flux, 1.0, atol=0.005)

        # Minimum flux must exhibit transit depth ~ 1 - (Rp/R*)^2 = 0.99
        min_flux = np.min(flux)
        assert min_flux < 0.995

    def test_robust_detrending_preserves_transit(self):
        """Detrending removes low frequency sine modulation without erasing transit depth."""
        time = np.linspace(0.0, 30.0, 1500)
        # Synthetic stellar variability (slow sinusoidal modulation)
        stellar_trend = 1.0 + 0.015 * np.sin(2.0 * np.pi * time / 7.0)
        flat_flux = np.ones_like(time)
        # Inject transit at t = 5.0
        in_transit = np.abs(time - 5.0) < 0.1
        flat_flux[in_transit] = 0.985  # 1.5% dip

        raw_flux = flat_flux * stellar_trend
        detrended = robust_detrend_flux(time, raw_flux, window_days=2.0)

        assert len(detrended) == len(raw_flux)
        # Baseline should be normalized around 1.0
        assert np.abs(np.median(detrended) - 1.0) < 0.005
        # Transit depression must remain present in detrended flux
        assert np.min(detrended[in_transit]) < 0.990

    def test_vectorized_bls_period_recovery(self):
        """Box Least Squares recovers true injected period P within 1% error tolerance."""
        true_period = 3.50
        true_t0 = 1.20
        time, flux, _ = generate_synthetic_light_curve(
            n_days=45.0,
            cadence_minutes=29.4,
            period_days=true_period,
            t0_days=true_t0,
            radius_ratio=0.09,
            snr_noise_sigma=0.0004,
            random_seed=42,
        )

        engine = BoxLeastSquaresEngine()
        test_periods = np.linspace(1.0, 10.0, 500)
        result = engine.search(time, flux, period_grid=test_periods)

        assert result.best_period > 0.0
        assert result.signal_residue > 0.0
        assert result.best_depth > 0.0
        # Accuracy check: detected period within 1.5% of true period
        rel_error = abs(result.best_period - true_period) / true_period
        assert rel_error < 0.015, f"BLS period error too high: {result.best_period:.4f} vs {true_period}"

    def test_dual_view_phase_folding_dimensions(self):
        """Dual-view phase folding accurately produces (2001,) Global View and (201,) Local View."""
        time, flux, _ = generate_synthetic_light_curve(
            n_days=30.0,
            cadence_minutes=29.4,
            period_days=3.0,
            t0_days=1.0,
            radius_ratio=0.08,
            snr_noise_sigma=0.0005,
            random_seed=99,
        )

        projection = compute_dual_view_projection(
            time=time,
            flux=flux,
            period_days=3.0,
            t0_days=1.0,
            duration_days=0.15,
            config=DEFAULT_PHOTOMETRY,
        )

        assert projection.global_view.shape == (2001,)
        assert projection.local_view.shape == (201,)
        assert projection.global_phases.shape == (2001,)
        assert projection.local_phases.shape == (201,)

        # Local view center (index 100) corresponds to transit center and should exhibit minimum flux
        center_idx = len(projection.local_view) // 2
        assert projection.local_view[center_idx] < 1.0
