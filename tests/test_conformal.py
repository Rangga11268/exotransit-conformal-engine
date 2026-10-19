"""Automated test suite for Milestone 3: Distribution-Free Split Conformal Prediction."""

import math
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

    def test_conformal_small_sample_infinite_quantile(self):
        """When calibration set is too small to mathematically satisfy nominal coverage (k > n),
        q_hat must be set to infinity and prediction sets must cover all classes."""
        # n = 5 samples, alpha = 0.05 -> ceil(6 * 0.95) = 6 > 5
        small_probs = np.array([
            [0.1, 0.9],
            [0.8, 0.2],
            [0.15, 0.85],
            [0.9, 0.1],
            [0.05, 0.95],
        ])
        small_labels = np.array([1, 0, 1, 0, 1])

        calibrator = SplitConformalCalibrator(nominal_error_rate=0.05)
        q_hat = calibrator.calibrate(small_probs, small_labels)

        assert np.isinf(q_hat), f"Expected q_hat=inf when k > n, got {q_hat}"

        test_p = np.array([[0.01, 0.99], [0.99, 0.01]])
        sets = calibrator.predict_set(test_p)
        assert len(sets) == 2
        for s in sets:
            assert s.status == STATUS_AMBIGUOUS
            assert set(s.prediction_set) == {"FALSE_POSITIVE", "PLANET"}

    def test_conformal_strict_input_validation(self):
        """Rejects illegal labels (e.g. -1), NaNs, and probabilities outside [0, 1]."""
        calibrator = SplitConformalCalibrator(nominal_error_rate=0.05)

        # Illegal label -1
        with pytest.raises(ValueError, match="strictly binary"):
            calibrator.calibrate(np.array([[0.5, 0.5]]), np.array([-1]))

        # NaN probabilities
        with pytest.raises(ValueError, match="finite"):
            calibrator.calibrate(np.array([[np.nan, 0.5]]), np.array([0]))

        # Out-of-bounds probability > 1.0
        with pytest.raises(ValueError, match="in \\[0.0, 1.0\\]"):
            calibrator.calibrate(np.array([[1.5, -0.5]]), np.array([0]))

        # Row sum does not equal 1.0
        with pytest.raises(ValueError, match="must sum to 1.0"):
            calibrator.calibrate(np.array([[0.8, 0.8]]), np.array([0]))

        calibrated_instance = SplitConformalCalibrator(nominal_error_rate=0.05)
        calibrated_instance.calibrate(self.cal_probs, self.cal_labels)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            calibrated_instance.predict_set(np.array([[0.2, 0.2]]))

    def test_conformal_evaluate_validation(self):
        """evaluate() must strictly validate inputs identically to calibrate()."""
        calibrator = SplitConformalCalibrator(nominal_error_rate=0.05)
        calibrator.calibrate(self.cal_probs, self.cal_labels)

        # Illegal label -1 in evaluate
        with pytest.raises(ValueError, match="strictly binary"):
            calibrator.evaluate(np.array([[0.5, 0.5]]), np.array([-1]))

        # NaN probabilities in evaluate
        with pytest.raises(ValueError, match="finite"):
            calibrator.evaluate(np.array([[np.nan, 0.5]]), np.array([0]))

        # Dimension mismatch
        with pytest.raises(ValueError, match="Lengths mismatch"):
            calibrator.evaluate(np.array([[0.5, 0.5], [0.8, 0.2]]), np.array([0]))

    def test_conformal_infinite_quantile_persistence(self, tmp_path):
        """Calibrator with q_hat=inf must persist and reload across standard JSON without errors."""
        calibrator = SplitConformalCalibrator(nominal_error_rate=0.05)
        # Small sample triggers q_hat = inf
        calibrator.calibrate(np.array([[0.1, 0.9], [0.8, 0.2]]), np.array([1, 0]))
        assert math.isinf(calibrator.q_hat)

        json_file = str(tmp_path / "infinite_calibration.json")
        calibrator.save(json_file)

        # Read raw json to ensure it's valid standard JSON
        with open(json_file, "r") as f:
            raw_text = f.read()
            assert "Infinity" in raw_text

        # Reload and check
        reloaded = SplitConformalCalibrator.load(json_file)
        assert math.isinf(reloaded.q_hat)
        assert reloaded.alpha == 0.05

    def test_candidate_ids_and_snrs_length_validation(self):
        """predict_set must reject candidate_ids or snrs with mismatched length."""
        calibrator = SplitConformalCalibrator(nominal_error_rate=0.05)
        calibrator.calibrate(self.cal_probs, self.cal_labels)

        probs = np.array([[0.8, 0.2], [0.3, 0.7]])  # 2 candidates

        # candidate_ids mismatch (1 ID for 2 rows)
        with pytest.raises(ValueError, match="candidate_ids length .* must match probabilities rows"):
            calibrator.predict_set(probs, candidate_ids=["KOI-001"])

        # snrs mismatch (3 SNRs for 2 rows)
        with pytest.raises(ValueError, match="snrs length .* must match probabilities rows"):
            calibrator.predict_set(probs, snrs=[15.2, 8.4, 22.1])

    def test_conformal_prediction_set_to_dict_infinite_and_finite(self):
        """Verify ConformalPredictionSet.to_dict handles finite and infinite quantiles safely."""
        # 1. Finite quantile
        ps_finite = ConformalPredictionSet(
            candidate_id="KOI-101",
            prediction_set=["PLANET"],
            p_false_positive=0.02,
            p_planet=0.98,
            quantile_threshold=0.85231,
            status=STATUS_SINGLETON_PLANET,
            set_size=1,
            is_singleton=True,
            snr=18.5,
        )
        d_fin = ps_finite.to_dict()
        assert d_fin["quantile_threshold"] == 0.8523

        # 2. Infinite quantile (must serialize to 'Infinity' string, not crash round())
        ps_inf = ConformalPredictionSet(
            candidate_id="KOI-102",
            prediction_set=["FALSE_POSITIVE", "PLANET"],
            p_false_positive=0.5,
            p_planet=0.5,
            quantile_threshold=float("inf"),
            status=STATUS_AMBIGUOUS,
            set_size=2,
            is_singleton=False,
            snr=None,
        )
        d_inf = ps_inf.to_dict()
        assert d_inf["quantile_threshold"] == "Infinity"

    def test_calibrator_load_alpha_validation(self, tmp_path):
        """Verify that loading calibration state validates expected runtime alpha."""
        cal = SplitConformalCalibrator(nominal_error_rate=0.05)
        cal.calibrate(self.cal_probs, self.cal_labels)
        save_path = str(tmp_path / "cal_alpha_test.json")
        cal.save(save_path)

        # Loading with matching expected_alpha succeeds
        loaded_ok = SplitConformalCalibrator.load(save_path, expected_alpha=0.05)
        assert loaded_ok.alpha == 0.05

        # Loading with mismatched expected_alpha raises ValueError
        with pytest.raises(ValueError, match="Calibration alpha mismatch"):
            SplitConformalCalibrator.load(save_path, expected_alpha=0.10)

    def test_round_trip_persistence_finite_quantile(self, tmp_path):
        """Verify save -> load -> predict consistency when q_hat is finite."""
        cal = SplitConformalCalibrator(nominal_error_rate=0.05)
        cal.calibrate(self.cal_probs, self.cal_labels)
        assert np.isfinite(cal.q_hat)

        test_probs = np.array([[0.1, 0.9], [0.8, 0.2], [0.45, 0.55]])
        preds_before = [p.to_dict() for p in cal.predict_set(test_probs)]

        save_file = str(tmp_path / "finite_cal.json")
        cal.save(save_file)

        loaded_cal = SplitConformalCalibrator.load(save_file)
        assert loaded_cal.alpha == cal.alpha
        assert loaded_cal.q_hat == cal.q_hat
        preds_after = [p.to_dict() for p in loaded_cal.predict_set(test_probs)]

        assert preds_before == preds_after

    def test_round_trip_persistence_infinite_quantile(self, tmp_path):
        """Verify save -> load -> predict consistency when q_hat is infinity."""
        # N=5, alpha=0.01 -> k = ceil(6 * 0.99) = 6 > 5 -> q_hat = inf
        cal = SplitConformalCalibrator(nominal_error_rate=0.01)
        small_probs = np.array([[0.2, 0.8], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.1, 0.9]])
        small_labels = np.array([1, 1, 0, 1, 1])
        q = cal.calibrate(small_probs, small_labels)
        assert math.isinf(q)

        test_probs = np.array([[0.1, 0.9], [0.8, 0.2]])
        preds_before = [p.to_dict() for p in cal.predict_set(test_probs)]

        save_file = str(tmp_path / "infinite_cal.json")
        cal.save(save_file)

        # Confirm JSON serialization contains string "Infinity"
        with open(save_file, "r") as f:
            content = f.read()
        assert '"Infinity"' in content

        loaded_cal = SplitConformalCalibrator.load(save_file)
        assert math.isinf(loaded_cal.q_hat)
        preds_after = [p.to_dict() for p in loaded_cal.predict_set(test_probs)]

        assert preds_before == preds_after
        for p in preds_after:
            assert p["prediction_set"] == ["FALSE_POSITIVE", "PLANET"]

    def test_alpha_mutation_invalidates_quantile_threshold(self):
        """Mutating calibrator alpha after calibration must invalidate q_hat and prevent silent stale reuse."""
        cal = SplitConformalCalibrator(nominal_error_rate=0.05)
        cal.calibrate(self.cal_probs, self.cal_labels)
        assert cal.q_hat is not None

        # Change alpha
        cal.alpha = 0.10
        assert cal.alpha == 0.10
        assert cal.q_hat is None

        # Predicting without recalibrating raises RuntimeError
        test_probs = np.array([[0.1, 0.9]])
        with pytest.raises(RuntimeError, match="Calibrator has not been calibrated"):
            cal.predict_set(test_probs)

    def test_invalid_alpha_bounds_rejected(self):
        """Alpha must be strictly within (0, 1)."""
        with pytest.raises(ValueError, match="nominal_error_rate must be in"):
            SplitConformalCalibrator(nominal_error_rate=0.0)
        with pytest.raises(ValueError, match="nominal_error_rate must be in"):
            SplitConformalCalibrator(nominal_error_rate=1.0)
        with pytest.raises(ValueError, match="nominal_error_rate must be in"):
            SplitConformalCalibrator(nominal_error_rate=-0.05)

    def test_input_guards_shape_mismatch_and_non_finite(self):
        """Calibrator rejects NaN, Inf, non-summing probabilities, and mismatched lengths."""
        cal = SplitConformalCalibrator(nominal_error_rate=0.05)
        cal.calibrate(self.cal_probs, self.cal_labels)

        # 1. NaN in probabilities
        with pytest.raises(ValueError, match="probabilities must contain only finite numbers"):
            cal.predict_set(np.array([[np.nan, 0.5]]))

        # 2. Infinite in probabilities
        with pytest.raises(ValueError, match="probabilities must contain only finite numbers"):
            cal.predict_set(np.array([[float("inf"), 0.5]]))

        # 3. Probabilities do not sum to 1.0
        with pytest.raises(ValueError, match="must sum to 1.0"):
            cal.predict_set(np.array([[0.2, 0.2]]))

        # 4. Candidate IDs length mismatch
        with pytest.raises(ValueError, match="candidate_ids length"):
            cal.predict_set(np.array([[0.1, 0.9]]), candidate_ids=["KOI-1", "KOI-2"])

        # 5. SNR length mismatch
        with pytest.raises(ValueError, match="snrs length"):
            cal.predict_set(np.array([[0.1, 0.9]]), snrs=[15.0, 20.0])



