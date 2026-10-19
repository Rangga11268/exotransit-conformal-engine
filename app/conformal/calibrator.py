"""Distribution-Free Split Conformal Prediction Calibrator.

Provides rigorous mathematical finite-sample guarantees:
P(Y_true in C(X)) >= 1 - alpha
Without relying on distributional or asymptotic normality assumptions.
"""

from dataclasses import dataclass
import json
import os
from typing import List, Dict, Optional, Any
import numpy as np


LABEL_NAMES = ["FALSE_POSITIVE", "PLANET"]

# Status definitions aligning with PRD.md R3
STATUS_SINGLETON_PLANET = "SINGLETON_PLANET"
STATUS_SINGLETON_FP = "SINGLETON_FALSE_POSITIVE"
STATUS_AMBIGUOUS = "AMBIGUOUS_REVIEW_REQUIRED"


@dataclass(frozen=True)
class ConformalPredictionSet:
    """Rigorous prediction set with finite-sample coverage guarantees."""
    candidate_id: str
    prediction_set: List[str]
    p_false_positive: float
    p_planet: float
    quantile_threshold: float
    status: str  # SINGLETON_PLANET, SINGLETON_FALSE_POSITIVE, AMBIGUOUS_REVIEW_REQUIRED
    set_size: int
    is_singleton: bool
    snr: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "prediction_set": self.prediction_set,
            "p_planet": round(self.p_planet, 4),
            "p_false_positive": round(self.p_false_positive, 4),
            "quantile_threshold": round(self.quantile_threshold, 4),
            "status": self.status,
            "set_size": self.set_size,
            "is_singleton": self.is_singleton,
            "snr": round(self.snr, 2) if self.snr is not None else None,
        }


class SplitConformalCalibrator:
    """Split Conformal Predictor for exoplanet transit classification.
    
    Computes non-conformity scores:
        s_i = 1 - P(Y_i | X_i)
    and derives the finite-sample adjusted quantile:
        q_hat = Quantile({s_i}, ceil((n + 1)(1 - alpha)) / n)
    """

    def __init__(self, nominal_error_rate: float = 0.05):
        """
        Parameters
        ----------
        nominal_error_rate : float
            Alpha error rate (default 0.05 guarantees >= 95% coverage).
        """
        if not (0.0 < nominal_error_rate < 1.0):
            raise ValueError(f"nominal_error_rate must be in (0, 1), got {nominal_error_rate}")
        self.alpha = float(nominal_error_rate)
        self.q_hat: Optional[float] = None
        self.calibration_scores: Optional[np.ndarray] = None
        self.n_calibration: int = 0
        self.source: Optional[str] = None

    @property
    def target_coverage(self) -> float:
        """Target nominal mathematical coverage (1 - alpha)."""
        return 1.0 - self.alpha

    def calibrate(self, probabilities: np.ndarray, ground_truth: np.ndarray) -> float:
        """Calibrate non-conformity quantile q_hat on held-out calibration split.
        
        Parameters
        ----------
        probabilities : np.ndarray
            Array of shape (n, 2) containing [P(FP), P(Planet)].
        ground_truth : np.ndarray
            Array of shape (n,) containing binary labels (0 or 1).
            
        Returns
        -------
        float : The finite-sample calibrated quantile q_hat.
        """
        probs = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(ground_truth, dtype=np.int64)

        if probs.ndim != 2 or probs.shape[1] != 2:
            raise ValueError(f"probabilities must have shape (n, 2), got {probs.shape}")
        if len(probs) != len(labels):
            raise ValueError(f"Lengths mismatch: {len(probs)} vs {len(labels)}")

        n = len(labels)
        if n == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Compute true class probabilities
        true_class_probs = probs[np.arange(n), labels]
        # Non-conformity scores: s_i = 1 - P(Y_i | X_i)
        scores = 1.0 - true_class_probs
        self.calibration_scores = np.sort(scores)
        self.n_calibration = n

        # Conformal finite-sample quantile index: ceil((n + 1) * (1 - alpha)) / n
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(1.0, max(0.0, level))

        # Quantile with higher method to ensure finite sample guarantee
        self.q_hat = float(np.quantile(self.calibration_scores, level, method="higher"))
        return self.q_hat

    def predict_set(
        self,
        probabilities: np.ndarray,
        candidate_ids: Optional[List[str]] = None,
        snrs: Optional[List[float]] = None
    ) -> List[ConformalPredictionSet]:
        """Construct conformal prediction sets for test inputs.
        
        Parameters
        ----------
        probabilities : np.ndarray
            Array of shape (m, 2) containing [P(FP), P(Planet)].
        candidate_ids : List[str], optional
            Identifiers for each candidate.
        snrs : List[float], optional
            Signal-to-noise ratio for each candidate.
            
        Returns
        -------
        List[ConformalPredictionSet] : Calibrated prediction sets.
        """
        if self.q_hat is None:
            raise RuntimeError("Calibrator has not been calibrated. Call calibrate() first.")

        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.ndim == 1:
            probs = probs.reshape(1, -1)
        if probs.shape[1] != 2:
            raise ValueError(f"Expected shape (*, 2), got {probs.shape}")

        m = len(probs)
        if candidate_ids is None:
            candidate_ids = [f"KOI-{i+1:04d}" for i in range(m)]

        cutoff = 1.0 - self.q_hat
        results = []

        for i in range(m):
            p_fp = float(probs[i, 0])
            p_planet = float(probs[i, 1])
            cid = candidate_ids[i]
            snr_val = snrs[i] if (snrs is not None and i < len(snrs)) else None

            # Label y is included if: P(y | X) >= 1 - q_hat
            prediction_set = []
            if p_fp >= cutoff:
                prediction_set.append("FALSE_POSITIVE")
            if p_planet >= cutoff:
                prediction_set.append("PLANET")

            # If neither class exceeds cutoff (uncertain/boundary candidate where model confidence is split),
            # this represents maximum uncertainty -> both classes included for manual review
            if len(prediction_set) == 0:
                prediction_set = ["FALSE_POSITIVE", "PLANET"]

            # Determine routing status
            k = len(prediction_set)
            if k == 1:
                if prediction_set[0] == "PLANET":
                    status = STATUS_SINGLETON_PLANET
                else:
                    status = STATUS_SINGLETON_FP
                is_singleton = True
            else:
                status = STATUS_AMBIGUOUS
                is_singleton = False

            results.append(
                ConformalPredictionSet(
                    candidate_id=cid,
                    prediction_set=prediction_set,
                    p_false_positive=p_fp,
                    p_planet=p_planet,
                    quantile_threshold=self.q_hat,
                    status=status,
                    set_size=k,
                    is_singleton=is_singleton,
                    snr=snr_val
                )
            )

        return results

    def evaluate(
        self,
        probabilities: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """Evaluates empirical coverage, singleton efficiency, and average set size.
        
        Returns
        -------
        dict with empirical_coverage, singleton_rate, ambiguity_rate, avg_set_size, target_coverage.
        """
        if self.q_hat is None:
            raise RuntimeError("Calibrator is not calibrated.")

        probs = np.asarray(probabilities, dtype=np.float64)
        labels = np.asarray(ground_truth, dtype=np.int64)
        m = len(labels)

        pred_sets = self.predict_set(probs)
        covered_count = 0
        singleton_count = 0
        ambiguous_count = 0
        total_size = 0

        for i, ps in enumerate(pred_sets):
            true_label_name = LABEL_NAMES[labels[i]]
            if true_label_name in ps.prediction_set:
                covered_count += 1
            if ps.is_singleton:
                singleton_count += 1
            else:
                ambiguous_count += 1
            total_size += ps.set_size

        empirical_coverage = float(covered_count / m) if m > 0 else 0.0
        singleton_rate = float(singleton_count / m) if m > 0 else 0.0
        ambiguity_rate = float(ambiguous_count / m) if m > 0 else 0.0
        avg_set_size = float(total_size / m) if m > 0 else 0.0

        return {
            "empirical_coverage": empirical_coverage,
            "target_coverage": self.target_coverage,
            "nominal_error_rate": self.alpha,
            "quantile_q_hat": self.q_hat,
            "singleton_rate": singleton_rate,
            "ambiguity_rate": ambiguity_rate,
            "avg_set_size": avg_set_size,
            "n_samples": m
        }

    def save(self, filepath: str) -> None:
        """Persists calibrated state to JSON."""
        data = {
            "alpha": self.alpha,
            "q_hat": self.q_hat,
            "n_calibration": self.n_calibration,
            "calibration_scores": self.calibration_scores.tolist() if self.calibration_scores is not None else None
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str, expected_alpha: Optional[float] = None) -> "SplitConformalCalibrator":
        """Loads calibrator from saved JSON file, optionally verifying expected alpha."""
        with open(filepath, "r") as f:
            data = json.load(f)
        saved_alpha = float(data["alpha"])
        if expected_alpha is not None and abs(saved_alpha - expected_alpha) > 1e-4:
            raise ValueError(
                f"Calibration alpha mismatch: file contains alpha={saved_alpha} "
                f"(coverage {1-saved_alpha:.1%}), expected alpha={expected_alpha} ({1-expected_alpha:.1%})"
            )
        cal = cls(nominal_error_rate=saved_alpha)
        cal.q_hat = data["q_hat"]
        cal.n_calibration = data["n_calibration"]
        if data.get("calibration_scores"):
            cal.calibration_scores = np.array(data["calibration_scores"], dtype=np.float64)
        return cal
