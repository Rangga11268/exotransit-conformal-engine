"""Automated test suite for Milestone 3: Distribution-Free Split Conformal Prediction."""

import os
import pytest
import numpy as np
from app.conformal.calibrator import (
    SplitConformalCalibrator,
    ConformalPredictionSet,
    STATUS_SINGLETON_PLANET,
    STATUS_SINGLETON_FP,
    STATUS_AMBIGUOUS,
)


class TestMilestone3Conformal:
    """Test suite for finite-sample conformal prediction layer."""

    @pytest.fixture(autouse=True)
    def setup_synthetic_data(self):
        """Generates realistic calibrated probabilities for testing."""
        rng = np.random.RandomState(42)
        n_total = 800

        # Ground truth labels: 0 = False Positive, 1 = Planet
        y_all = rng.binomial(1, 0.5, size=n_total)

        # Generate realistic model probabilities
        probs = np.zeros((n_total, 2), dtype=np.float64)
        for i in range(n_total):
            true_y = y_all[i]
            # Well-behaved model with ~90% accuracy and occasional confusion
            if true_y == 1:
                p1 = rng.beta(6.0, 1.5) # skewed towards high probability
            else:
                p1 = rng.beta(1.5, 6.0) # skewed towards low probability
            probs[i, 1] = p1
            probs[i, 0] = 1.0 - p1

        # Split 50% calibration, 50% test
        self.cal_probs = probs[:400]
        self.cal_labels = y_all[:400]
        self.test_probs = probs[400:]
        self.test_labels = y_all[400:]

    def test_conformal_finite_sample_coverage(self):
        """Mathematical property: empirical coverage on test set must satisfy >= 1 - alpha."""
        alpha = 0.05
        calibrator = SplitConformalCalibrator(nominal_error_rate=alpha)
        q_hat = calibrator.calibrate(self.cal_probs, self.cal_labels)

        assert 0.0 < q_hat < 1.0, f"q_hat must be in (0, 1), got {q_hat}"

        eval_res = calibrator.evaluate(self.test_probs, self.test_labels)
        empirical_cov = eval_res["empirical_coverage"]

        # Finite-sample coverage bound (guarantees >= 95% nominal coverage)
        assert empirical_cov >= 0.942, (
            f"Coverage violated! Expected >= 0.942 at alpha=0.05, got {empirical_cov:.3f}"
        )
        assert eval_res["singleton_rate"] > 0.70, (
            f"Singleton efficiency too low: {eval_res['singleton_rate']:.3f}"
        )
        assert 1.0 <= eval_res["avg_set_size"] <= 2.0

    def test_conformal_status_routing(self):
        """Verifies prediction sets and routing statuses for sharp vs ambiguous candidates."""
        calibrator = SplitConformalCalibrator(nominal_error_rate=0.05)
        calibrator.calibrate(self.cal_probs, self.cal_labels)

        # Case 1: Ultra confident Planet
        p_clean_planet = np.array([[0.01, 0.99]])
        set_planet = calibrator.predict_set(p_clean_planet, candidate_ids=["KOI-001"])[0]
        assert set_planet.status == STATUS_SINGLETON_PLANET
        assert set_planet.prediction_set == ["PLANET"]
        assert set_planet.is_singleton is True

        # Case 2: Ultra confident False Positive
        p_clean_fp = np.array([[0.99, 0.01]])
        set_fp = calibrator.predict_set(p_clean_fp, candidate_ids=["KOI-002"])[0]
        assert set_fp.status == STATUS_SINGLETON_FP
        assert set_fp.prediction_set == ["FALSE_POSITIVE"]
        assert set_fp.is_singleton is True

        # Case 3: Ambiguous borderline signal (e.g. low SNR transit)
        p_ambig = np.array([[0.52, 0.48]])
        set_ambig = calibrator.predict_set(p_ambig, candidate_ids=["KOI-003"])[0]
        assert set_ambig.status == STATUS_AMBIGUOUS
        assert "PLANET" in set_ambig.prediction_set
        assert "FALSE_POSITIVE" in set_ambig.prediction_set
        assert set_ambig.is_singleton is False

    def test_conformal_persistence(self, tmp_path):
        """Calibrator must persist quantile to JSON and reload with zero drift."""
        calibrator = SplitConformalCalibrator(nominal_error_rate=0.05)
        q_hat = calibrator.calibrate(self.cal_probs, self.cal_labels)

        json_file = str(tmp_path / "calibration.json")
        calibrator.save(json_file)
        assert os.path.exists(json_file)

        # Reload
        reloaded = SplitConformalCalibrator.load(json_file)
        assert reloaded.alpha == 0.05
        assert abs(reloaded.q_hat - q_hat) < 1e-6

        # Predictions match identically
        preds1 = calibrator.predict_set(self.test_probs[:10])
        preds2 = reloaded.predict_set(self.test_probs[:10])
        for p1, p2 in zip(preds1, preds2):
            assert p1.prediction_set == p2.prediction_set
            assert p1.status == p2.status

    def test_conformal_validation_guards(self):
        """Calibrator must reject invalid alpha rates and uncalibrated predictions."""
        with pytest.raises(ValueError):
            SplitConformalCalibrator(nominal_error_rate=0.0)

        with pytest.raises(ValueError):
            SplitConformalCalibrator(nominal_error_rate=1.0)

        uncal = SplitConformalCalibrator(nominal_error_rate=0.05)
        with pytest.raises(RuntimeError):
            uncal.predict_set(np.array([[0.5, 0.5]]))
