"""Automated test suite for Milestone 2: Dual-Branch Exoplanet Classifier."""

import time
import os
import pytest
import numpy as np
from app.photometry.simulator import simulate_transit_light_curve, inject_stellar_noise
from app.photometry.phase_fold import generate_dual_views
from app.models.feature_extractor import DualBranchFeatureExtractor
from app.models.classifier import DualBranchClassifier, generate_synthetic_dataset


class TestMilestone2Models:
    """Test suite for dual-branch feature extractor and lightweight classifier."""

    @pytest.fixture(autouse=True)
    def setup_data(self):
        """Prepare sample clean planet transit and false positive binary light curves."""
        self.cadence_days = 29.4 / (24.0 * 60.0)
        self.time = np.arange(0.0, 30.0, self.cadence_days)
        self.period = 3.52
        self.epoch = 0.8
        self.duration = 0.12

        # 1. Clear exoplanet (Mandel-Agol U-shape, limb-darkened, SNR ~ 20)
        flux_planet = simulate_transit_light_curve(
            time=self.time,
            period=self.period,
            epoch=self.epoch,
            duration=self.duration,
            radius_ratio=0.04,
            u1=0.3,
            u2=0.2,
            impact_parameter=0.2
        )
        self.flux_planet = inject_stellar_noise(flux_planet, noise_std=2e-4, random_seed=42)
        self.planet_views = generate_dual_views(
            self.time, self.flux_planet, self.period, self.epoch, self.duration
        )

        # 2. Eclipsing Binary (grazing V-shape with 50% secondary eclipse at phase 0.5)
        flux_pri = simulate_transit_light_curve(
            time=self.time,
            period=self.period,
            epoch=self.epoch,
            duration=self.duration * 1.5,
            radius_ratio=0.15,
            u1=0.1,
            u2=0.1,
            impact_parameter=0.85
        )
        flux_sec = simulate_transit_light_curve(
            time=self.time,
            period=self.period,
            epoch=self.epoch + 0.5 * self.period,
            duration=self.duration * 1.5,
            radius_ratio=0.10,
            u1=0.1,
            u2=0.1,
            impact_parameter=0.85
        )
        self.flux_eb = inject_stellar_noise(flux_pri + (flux_sec - 1.0), noise_std=3e-4, random_seed=43)
        self.eb_views = generate_dual_views(
            self.time, self.flux_eb, self.period, self.epoch, self.duration * 1.5
        )

    def test_feature_extractor_shape_and_finite(self):
        """Feature extractor must output exact 43-dim array with finite numeric values."""
        extractor = DualBranchFeatureExtractor()
        feat = extractor.extract(
            global_view=self.planet_views.global_view,
            local_view=self.planet_views.local_view,
            extra_scalars={"period": self.period, "duration": self.duration, "depth": 0.0016, "sde": 15.0, "duty_cycle": 0.03}
        )

        assert isinstance(feat, np.ndarray)
        assert feat.shape == (43,)
        assert np.all(np.isfinite(feat)), "Feature vector contains NaN or Inf values!"

    def test_classifier_training_and_persistence(self, tmp_path):
        """Classifier should train, persist model to file, and reload identically."""
        model_file = str(tmp_path / "test_model.txt")
        clf = DualBranchClassifier(model_path=model_file)
        assert os.path.exists(model_file), "Model file was not created!"

        # Reload
        clf_reloaded = DualBranchClassifier(model_path=model_file)
        prob1 = clf.predict_proba(self.planet_views.global_view, self.planet_views.local_view)
        prob2 = clf_reloaded.predict_proba(self.planet_views.global_view, self.planet_views.local_view)

        np.testing.assert_allclose(prob1, prob2, atol=1e-5)

    def test_classifier_inference_latency(self):
        """Inference latency per candidate must be sub-10ms on CPU."""
        clf = DualBranchClassifier()
        
        # Warmup
        _ = clf.predict_proba(self.planet_views.global_view, self.planet_views.local_view)

        # Benchmark 20 iterations
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            probs = clf.predict_proba(self.planet_views.global_view, self.planet_views.local_view)
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            times.append(dt)

        avg_latency = np.mean(times)
        p99_latency = np.percentile(times, 99)
        assert p99_latency < 15.0, f"P99 latency too high: {p99_latency:.2f} ms > 15 ms target"
        assert len(probs) == 2
        assert abs(np.sum(probs) - 1.0) < 1e-4

    def test_classifier_discrimination(self):
        """Model must give higher probability of Planet to genuine transit than to EB."""
        clf = DualBranchClassifier()
        prob_planet = clf.predict_proba(self.planet_views.global_view, self.planet_views.local_view)
        prob_eb = clf.predict_proba(self.eb_views.global_view, self.eb_views.local_view)

        # P(Planet) is index 1
        p_planet_genuine = prob_planet[1]
        p_planet_eb = prob_eb[1]

        assert p_planet_genuine > p_planet_eb, (
            f"Planet score ({p_planet_genuine:.3f}) should exceed Eclipsing Binary score ({p_planet_eb:.3f})"
        )
