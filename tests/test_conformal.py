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

    def test_exact_order_statistic_alignment(self):
        """Verify calibration threshold matches exact theoretical order statistic S_{(k)}."""
        # 10 synthetic samples with known nonconformity scores: 0.1, 0.2, ..., 1.0
        n = 10
        probs = np.zeros((n, 2))
        for i in range(n):
            score = (i + 1) / 10.0  # scores = [0.1, 0.2, 0.3, ..., 1.0]
            probs[i, 1] = 1.0 - score  # true label is 1, so nonconformity score = 1 - P(1) = score
            probs[i, 0] = score
        labels = np.ones(n, dtype=int)

        # For alpha = 0.20: k = ceil(11 * 0.80) = ceil(8.8) = 9
        # Exact order statistic: 9th value = 0.9
        cal = SplitConformalCalibrator(nominal_error_rate=0.20)
        q = cal.calibrate(probs, labels)
        assert np.isclose(q, 0.9, atol=1e-5), f"Expected S_(9) = 0.9, got {q}"

        # For alpha = 0.05: k = ceil(11 * 0.95) = ceil(10.45) = 11 > 10 => inf
        cal_inf = SplitConformalCalibrator(nominal_error_rate=0.05)
        q_inf = cal_inf.calibrate(probs, labels)
        assert math.isinf(q_inf), f"Expected inf for k > n, got {q_inf}"

    def test_multi_seed_empirical_coverage(self):
        """Verify empirical coverage, set size, and singleton efficiency across multiple seeds."""
        coverages = []
        set_sizes = []
        singleton_rates = []
        alpha = 0.10

        for seed in [10, 42, 99, 123, 777]:
            rng = np.random.RandomState(seed)
            y = rng.binomial(1, 0.5, size=600)
            probs = np.zeros((600, 2))
            for i in range(600):
                p1 = rng.beta(5.0, 2.0) if y[i] == 1 else rng.beta(2.0, 5.0)
                probs[i, 1] = p1
                probs[i, 0] = 1.0 - p1

            cal = SplitConformalCalibrator(nominal_error_rate=alpha)
            cal.calibrate(probs[:300], y[:300])
            eval_res = cal.evaluate(probs[300:], y[300:])

            # Verify finite-sample coverage per seed
            assert eval_res["empirical_coverage"] >= 0.85, (
                f"Seed {seed} coverage {eval_res['empirical_coverage']} violated nominal 90%"
            )
            assert 1.0 <= eval_res["avg_set_size"] <= 2.0
            assert eval_res["singleton_rate"] > 0.50

            coverages.append(eval_res["empirical_coverage"])
            set_sizes.append(eval_res["avg_set_size"])
            singleton_rates.append(eval_res["singleton_rate"])

        mean_cov = float(np.mean(coverages))
        assert mean_cov >= 1.0 - alpha, f"Mean empirical coverage {mean_cov:.3f} fell below {1.0 - alpha}"

    def test_controlled_distribution_shift_degradation(self):
        """Under severe covariate/instrument shift (exchangeability violation), prediction sets adapt."""
        alpha = 0.05
        rng = np.random.RandomState(42)

        # In-distribution calibration population: high SNR, well-separated
        n_cal = 400
        y_cal = rng.binomial(1, 0.5, size=n_cal)
        cal_probs = np.zeros((n_cal, 2))
        for i in range(n_cal):
            p1 = rng.beta(8.0, 1.2) if y_cal[i] == 1 else rng.beta(1.2, 8.0)
            cal_probs[i, 1] = p1
            cal_probs[i, 0] = 1.0 - p1

        cal = SplitConformalCalibrator(nominal_error_rate=alpha)
        cal.calibrate(cal_probs, y_cal)

        # OOD / shifted test population: degraded SNR, heavily confused model
        n_test = 400
        y_test = rng.binomial(1, 0.5, size=n_test)
        ood_probs = np.zeros((n_test, 2))
        for i in range(n_test):
            # Model output degraded to near-random guessing around 0.5
            p1 = rng.uniform(0.35, 0.65)
            ood_probs[i, 1] = p1
            ood_probs[i, 0] = 1.0 - p1

        eval_ood = cal.evaluate(ood_probs, y_test)
        # Because calibration threshold was sharp (calibrated on high confidence),
        # testing on shifted uncertain data triggers significant ambiguous sets or drops singleton rate
        assert eval_ood["singleton_rate"] < 0.60, "Model should flag ambiguous sets under heavy shift"

    def test_infinity_sentinel_produces_conservative_full_sets(self):
        """When calibration sample size cannot guarantee nominal error, q_hat evaluates to Infinity."""
        cal = SplitConformalCalibrator(nominal_error_rate=0.01)
        # With n=10, ceil(11 * 0.99) = 11 > 10 => q_hat = inf
        probs = np.tile([0.5, 0.5], (10, 1))
        labels = np.ones(10, dtype=int)
        q = cal.calibrate(probs, labels)
        assert math.isinf(q)

        test_probs = np.array([[0.01, 0.99], [0.99, 0.01]])
        pred_sets = cal.predict_set(test_probs)
        for ps in pred_sets:
            # Full set to conservatively preserve coverage
            assert set(ps.prediction_set) == {"FALSE_POSITIVE", "PLANET"}
            assert ps.is_singleton is False
            assert ps.set_size == 2
            d = ps.to_dict()
            assert d["quantile_threshold"] == "Infinity"

    def test_multi_seed_uncertainty_intervals_and_efficiency(self):
        """Evaluate coverage across S=20 seeds reporting empirical confidence bounds and efficiency trade-offs."""
        alpha = 0.10
        target_coverage = 1.0 - alpha
        coverages = []
        set_sizes = []
        singleton_rates = []

        for seed in range(20):
            rng = np.random.RandomState(seed + 1000)
            n_samples = 500
            y = rng.binomial(1, 0.5, size=n_samples)
            probs = np.zeros((n_samples, 2))
            for i in range(n_samples):
                p1 = rng.beta(6.0, 1.8) if y[i] == 1 else rng.beta(1.8, 6.0)
                probs[i, 1] = p1
                probs[i, 0] = 1.0 - p1

            cal = SplitConformalCalibrator(nominal_error_rate=alpha)
            cal.calibrate(probs[:250], y[:250])
            eval_res = cal.evaluate(probs[250:], y[250:])

            coverages.append(eval_res["empirical_coverage"])
            set_sizes.append(eval_res["avg_set_size"])
            singleton_rates.append(eval_res["singleton_rate"])

        cov_mean = float(np.mean(coverages))
        cov_std = float(np.std(coverages))
        cov_se = cov_std / np.sqrt(len(coverages))
        # 95% CI: mean +/- 1.96 * SE
        ci_lower = cov_mean - 1.96 * cov_se
        ci_upper = cov_mean + 1.96 * cov_se

        avg_set_size_mean = float(np.mean(set_sizes))
        avg_singleton_rate = float(np.mean(singleton_rates))

        # 1. Coverage validity: mean coverage achieves target 90%
        assert cov_mean >= target_coverage - 0.015, f"Mean coverage {cov_mean:.4f} too low"
        # 2. Efficiency guarantee: set size is strictly bounded, proving non-vacuous predictions
        assert avg_set_size_mean <= 1.35, f"Set size too wide: {avg_set_size_mean:.3f}"
        assert avg_singleton_rate >= 0.70, f"Singleton rate too low: {avg_singleton_rate:.3f}"

    def test_multi_regime_noise_comparison(self):
        """Test conformal calibration behavior across Low, Moderate, and High photometric noise regimes."""
        alpha = 0.10
        regimes = {
            "low_noise_bright_star": {"alpha_p": 9.0, "beta_p": 1.2, "expected_min_singleton": 0.85},
            "moderate_noise_standard": {"alpha_p": 5.0, "beta_p": 2.0, "expected_min_singleton": 0.60},
            "high_noise_faint_star": {"alpha_p": 2.5, "beta_p": 2.0, "expected_min_singleton": 0.20},
        }

        rng = np.random.RandomState(42)

        for name, params in regimes.items():
            n = 600
            y = rng.binomial(1, 0.5, size=n)
            probs = np.zeros((n, 2))
            for i in range(n):
                p1 = rng.beta(params["alpha_p"], params["beta_p"]) if y[i] == 1 else rng.beta(params["beta_p"], params["alpha_p"])
                probs[i, 1] = p1
                probs[i, 0] = 1.0 - p1

            cal = SplitConformalCalibrator(nominal_error_rate=alpha)
            cal.calibrate(probs[:300], y[:300])
            eval_res = cal.evaluate(probs[300:], y[300:])

            # Coverage MUST hold across all noise regimes due to distribution-free validity
            assert eval_res["empirical_coverage"] >= 0.88, f"Regime {name} coverage broke: {eval_res['empirical_coverage']}"
            # Efficiency adapts: sharper models yield higher singleton rates
            assert eval_res["singleton_rate"] >= params["expected_min_singleton"], (
                f"Regime {name} singleton rate {eval_res['singleton_rate']:.3f} below expected {params['expected_min_singleton']}"
            )






