"""
conftest.py
-----------
Shared fixtures for text-service tests.

Key design decisions:
  - BERT model is mocked — tests never require the actual model files.
  - Mock returns deterministic scores based on input content.
  - Suspicious phrase detection is tested with real logic (no mock needed).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient


# ── mock predict_fraud ────────────────────────────────────────────────────────

def make_mock_predict(fraud_score: float = 80.0):
    """
    Return a mock predict_fraud function that returns a fixed score.
    Override fraud_score per test class as needed.
    """
    def _predict(text: str) -> dict:
        score = fraud_score
        return {
            "fraud_score":     score,
            "label":           "FRAUD" if score >= 50 else "LEGITIMATE",
            "confidence":      score,
            "fraud_prob":      score,
            "legitimate_prob": round(100.0 - score, 1),
        }
    return _predict


def make_mock_predict_legitimate():
    return make_mock_predict(fraud_score=10.0)


def make_mock_predict_fraud():
    return make_mock_predict(fraud_score=90.0)


# ── FastAPI test client ───────────────────────────────────────────────────────

@pytest.fixture()
def client():
    """
    TestClient with model loading mocked.
    model_loaded=True, predict_fraud returns a fixed score of 80.
    """
    import main as main_module

    with patch("predict.load_model", return_value=None):
        with patch("predict.predict_fraud", side_effect=make_mock_predict(80.0)):
            from main import app
            app.state.model_loaded = True
            with TestClient(app) as c:
                yield c


@pytest.fixture()
def client_legitimate():
    """TestClient where predict_fraud returns a low (legitimate) score."""
    with patch("predict.load_model", return_value=None):
        with patch("predict.predict_fraud", side_effect=make_mock_predict_legitimate()):
            from main import app
            app.state.model_loaded = True
            with TestClient(app) as c:
                yield c


@pytest.fixture()
def client_unloaded():
    """TestClient where model failed to load — modelLoaded=False."""
    with patch("predict.load_model", side_effect=FileNotFoundError("model not found")):
        from main import app
        app.state.model_loaded = False
        with TestClient(app) as c:
            yield c


# ── sample payloads ───────────────────────────────────────────────────────────

FRAUD_PAYLOAD = {
    "claimId": "SIN-2026-001",
    "claimDescription": "Sudden failure with no warning. Machine was perfectly maintained.",
    "maintenanceReportText": "All checks passed. No anomalies detected before incident.",
}

LEGITIMATE_PAYLOAD = {
    "claimId": "SIN-2026-002",
    "claimDescription": "Gradual increase in vibration observed over 3 weeks before failure.",
    "maintenanceReportText": "Temperature readings showed steady rise from 65C to 89C over 23 days.",
}