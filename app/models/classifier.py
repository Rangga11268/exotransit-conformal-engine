"""Dual-Branch Exoplanet Classifier.

Combines multi-scale 1D morphological features from Global & Local Views with
astrophysical scalar indicators, producing calibrated probabilities [P(False Positive), P(Planet)].
Uses native LightGBM on Windows CPU with zero heavy C++ runtime dependencies (<20MB memory).
"""

import os
from typing import Tuple, List, Optional, Union
import numpy as np
import lightgbm as lgb
from app.photometry.bls import BLSResult
from app.photometry.simulator import simulate_transit_light_curve, inject_stellar_noise
from app.photometry.phase_fold import generate_dual_views
from app.models.feature_extractor import DualBranchFeatureExtractor


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "exoplanet_lgb_model.txt")


def generate_synthetic_dataset(n_samples: int = 600, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a balanced synthetic dataset of planets vs astrophysical false positives."""
    rng = np.random.RandomState(random_seed)
    extractor = DualBranchFeatureExtractor()

    features_list = []
    labels_list = []

    for i in range(n_samples):
        # 50% planets, 50% false positives
        is_planet = (i % 2 == 1)
        
        period = rng.uniform(1.0, 10.0)
        time_span = 15.0  # 15 days
        cadence_days = 29.4 / (24.0 * 60.0) # ~30 min Kepler cadence
        time = np.arange(0.0, time_span, cadence_days)
        epoch = rng.uniform(0.1, period * 0.8)

        
        if is_planet:
            # Genuine exoplanet: U-shape Mandel-Agol, no secondary eclipse
            # Covers rocky super-Earths to Jupiter gas giants (1.2% to 12.0% stellar radius)
            radius_ratio = rng.uniform(0.012, 0.12)
            duration = rng.uniform(0.05, 0.25)
            u1, u2 = 0.3, 0.2
            
            # Primary transit
            flux = simulate_transit_light_curve(
                time=time,
                period=period,
                epoch=epoch,
                duration=duration,
                radius_ratio=radius_ratio,
                u1=u1,
                u2=u2,
                impact_parameter=rng.uniform(0.0, 0.6) # flat bottom U-shape
            )
            # Modest noise
            noise_std = rng.uniform(1e-4, 5e-4)
            flux = inject_stellar_noise(flux, noise_std=noise_std, random_seed=rng.randint(0, 100000))
            labels_list.append(1)
            depth = radius_ratio ** 2
        else:
            # False Positive: either Eclipsing Binary or Stellar Noise / Variable Star
            fp_type = rng.choice(["eb", "noise", "v_shape"])
            if fp_type == "eb":
                # Eclipsing binary with noticeable secondary eclipse at phase 0.5
                primary_depth = rng.uniform(0.01, 0.06)
                sec_depth = primary_depth * rng.uniform(0.2, 0.7)
                duration = rng.uniform(0.08, 0.3)
                
                # Primary
                flux = simulate_transit_light_curve(
                    time=time,
                    period=period,
                    epoch=epoch,
                    duration=duration,
                    radius_ratio=np.sqrt(primary_depth),
                    u1=0.1,
                    u2=0.1,
                    impact_parameter=0.85 # grazing V-shape
                )
                # Secondary at phase 0.5
                sec_epoch = epoch + 0.5 * period
                flux_sec = simulate_transit_light_curve(
                    time=time,
                    period=period,
                    epoch=sec_epoch,
                    duration=duration,
                    radius_ratio=np.sqrt(sec_depth),
                    u1=0.1,
                    u2=0.1,
                    impact_parameter=0.85
                )
                flux = flux + (flux_sec - 1.0)
                flux = inject_stellar_noise(flux, noise_std=rng.uniform(2e-4, 8e-4), random_seed=rng.randint(0, 100000))
                depth = primary_depth

            elif fp_type == "v_shape":
                # High-impact grazing binary (V-shape without flat bottom)
                duration = rng.uniform(0.05, 0.2)
                flux = simulate_transit_light_curve(
                    time=time,
                    period=period,
                    epoch=epoch,
                    duration=duration,
                    radius_ratio=0.15,
                    u1=0.0,
                    u2=0.0,
                    impact_parameter=0.98 # extreme grazing V-shape
                )
                flux = inject_stellar_noise(flux, noise_std=rng.uniform(3e-4, 1e-3), random_seed=rng.randint(0, 100000))
                depth = 0.02

            else:
                # Pure sinusoidal stellar variability + instrumental drift
                flux = 1.0 + 0.003 * np.sin(2.0 * np.pi * time / rng.uniform(2.0, 7.0))
                flux = inject_stellar_noise(flux, noise_std=rng.uniform(5e-4, 1.5e-3), random_seed=rng.randint(0, 100000))
                duration = 0.1
                depth = 0.002

            labels_list.append(0)

        # Generate dual views
        views = generate_dual_views(time, flux, period=period, epoch=epoch, duration=duration)
        scalar_dict = {
            "period": period,
            "duration": duration,
            "depth": depth,
            "sde": depth / (np.std(flux) + 1e-6),
            "duty_cycle": duration / period
        }
        feat = extractor.extract(views.global_view, views.local_view, extra_scalars=scalar_dict)
        features_list.append(feat)

    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)
    return X, y


class DualBranchClassifier:
    """Production-grade classifier scoring exoplanet candidate probabilities."""

    def __init__(self, model_path: Optional[str] = DEFAULT_MODEL_PATH):
        self.extractor = DualBranchFeatureExtractor()
        self.model_path = model_path
        self.booster: Optional[lgb.Booster] = None
        
        if model_path and os.path.exists(model_path):
            self.booster = lgb.Booster(model_file=model_path)
        else:
            self.train_and_save(model_path)

    def train_and_save(self, save_path: Optional[str] = None, n_samples: int = 200) -> None:
        """Trains booster on synthetic astrophysics dataset and persists model."""
        X, y = generate_synthetic_dataset(n_samples=n_samples, random_seed=1337)
        train_data = lgb.Dataset(X, label=y)
        
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 15,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "verbose": -1,
            "min_data_in_leaf": 5,
            "n_jobs": 2
        }
        
        self.booster = lgb.train(
            params,
            train_data,
            num_boost_round=40
        )

        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.booster.save_model(save_path)

    def predict_proba_from_features(self, features: np.ndarray) -> np.ndarray:
        """Computes probabilities [P(FP), P(Planet)] from feature vector.
        
        Args:
            features: 1D array of shape (D,) or 2D array of shape (N, D).
            
        Returns:
            2D array of shape (N, 2) where col 0 is P(FP) and col 1 is P(Planet).
        """
        if self.booster is None:
            raise RuntimeError("Classifier model is not loaded or trained.")
            
        feat = np.atleast_2d(features)
        raw_prob_planet = self.booster.predict(feat) # shape (N,)
        raw_prob_planet = np.clip(raw_prob_planet, 1e-4, 1.0 - 1e-4)
        prob_fp = 1.0 - raw_prob_planet
        return np.column_stack([prob_fp, raw_prob_planet])

    def predict_proba(
        self,
        global_view: np.ndarray,
        local_view: np.ndarray,
        bls_result: Optional[BLSResult] = None,
        extra_scalars: Optional[dict] = None
    ) -> np.ndarray:
        """Computes probabilities [P(FP), P(Planet)] from raw dual-views.
        
        Returns:
            1D array of shape (2,): [P(False Positive), P(Planet)].
        """
        features = self.extractor.extract(
            global_view=global_view,
            local_view=local_view,
            bls_result=bls_result,
            extra_scalars=extra_scalars
        )
        proba_2d = self.predict_proba_from_features(features)
        return proba_2d[0]
