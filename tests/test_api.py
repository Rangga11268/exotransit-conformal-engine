"""Automated test suite for Milestone 4: FastAPI REST API Endpoints."""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.photometry.simulator import simulate_transit_light_curve, inject_stellar_noise


@pytest.fixture(scope="module")
def client():
    """Module-scoped FastAPI TestClient running lifecycle events."""
    with TestClient(app) as c:
        yield c


class TestMilestone4API:
    """End-to-end integration tests for FastAPI exoplanet detection service."""

    @pytest.fixture(autouse=True)
    def setup_payload(self):
        """Prepare sample light curve with Kepler-like sampling."""
        cadence_days = 29.4 / (24.0 * 60.0)
        self.time = np.arange(0.0, 20.0, cadence_days)
        self.period = 3.524
        self.epoch = 1.2
        self.duration = 0.15
        
        flux = simulate_transit_light_curve(
            time=self.time,
            period=self.period,
            epoch=self.epoch,
            duration=self.duration,
            radius_ratio=0.06,
            u1=0.35,
            u2=0.20,
            impact_parameter=0.3
        )
        self.flux = inject_stellar_noise(flux, noise_std=2e-4, random_seed=42)

    def test_root_endpoint(self, client):
        """GET / must return operational status."""
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["engine"] == "ExoTransit-Conformal"
        assert data["status"] == "operational"

    def test_health_endpoint(self, client):
        """GET /v1/health must report model loaded and calibrated status."""
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["is_calibrated"] is True
        assert data["active_q_hat"] is not None

    def test_transit_detection_pipeline(self, client):
        """POST /v1/transit/detect must execute complete pipeline with conformal sets."""
        payload = {
            "time": self.time.tolist(),
            "flux": self.flux.tolist(),
            "candidate_id": "KOI-TEST-777",
            "alpha": 0.05,
            "detrend": True
        }
        resp = client.post("/v1/transit/detect", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["candidate_id"] == "KOI-TEST-777"
        assert "bls_summary" in data
        assert abs(data["bls_summary"]["period_days"] - self.period) < 0.20
        assert data["estimated_radius_ratio"] > 0.0
        assert "probabilities" in data
        assert "PLANET" in data["probabilities"]
        assert "FALSE_POSITIVE" in data["probabilities"]
        
        conf = data["conformal_prediction"]
        assert "prediction_set" in conf
        assert "status" in conf
        assert conf["status"] in ["SINGLETON_PLANET", "SINGLETON_FALSE_POSITIVE", "AMBIGUOUS_REVIEW_REQUIRED"]
        assert data["processing_time_ms"] > 0.0

    def test_transit_detection_validation_errors(self, client):
        """Pipeline must gracefully reject mismatched or malformed inputs."""
        # 1. Length mismatch
        resp1 = client.post("/v1/transit/detect", json={"time": [1.0, 2.0], "flux": [1.0]})
        assert resp1.status_code == 400

        # 2. Too short (< 50 points)
        resp2 = client.post("/v1/transit/detect", json={"time": [float(i) for i in range(20)], "flux": [1.0]*20})
        assert resp2.status_code == 400

        # 3. Non-finite values
        resp3 = client.post("/v1/transit/detect", json={"time": [1.0]*60, "flux": [1.0]*59 + [None]})
        assert resp3.status_code == 422 # Pydantic schema validation error

    def test_recalibrate_endpoint(self, client):
        """POST /v1/conformal/calibrate must update error rate and return acknowledgment."""
        payload = {
            "alpha": 0.10,
            "n_synthetic_samples": 100
        }
        resp = client.post("/v1/conformal/calibrate", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "recalibrated_successfully"
        assert data["nominal_error_rate"] == 0.10
        assert data["target_coverage"] == 0.90
        assert 0.0 < data["quantile_q_hat"] < 1.0
