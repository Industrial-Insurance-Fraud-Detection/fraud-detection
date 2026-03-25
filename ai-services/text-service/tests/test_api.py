"""
tests/test_api.py

HTTP layer tests for text-service:
  - GET  /health
  - POST /analyze-text — schema contract, validation, edge cases
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FRAUD_PAYLOAD, LEGITIMATE_PAYLOAD


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient):
        body = client.get("/health").json()
        assert "status"      in body
        assert "service"     in body
        assert "modelLoaded" in body

    def test_status_ok_when_model_loaded(self, client: TestClient):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["modelLoaded"] is True

    def test_status_degraded_when_model_missing(self, client_unloaded: TestClient):
        body = client_unloaded.get("/health").json()
        assert body["status"] == "degraded"
        assert body["modelLoaded"] is False


# ── /analyze-text — validation ────────────────────────────────────────────────

class TestAnalyzeValidation:
    def test_missing_claim_description_422(self, client: TestClient):
        r = client.post("/analyze-text", json={
            "maintenanceReportText": "Some report text."
        })
        assert r.status_code == 422

    def test_missing_maintenance_report_422(self, client: TestClient):
        r = client.post("/analyze-text", json={
            "claimDescription": "Some claim text."
        })
        assert r.status_code == 422

    def test_empty_body_422(self, client: TestClient):
        r = client.post("/analyze-text", json={})
        assert r.status_code == 422

    def test_claim_id_is_optional(self, client: TestClient):
        payload = {
            "claimDescription": "Some claim.",
            "maintenanceReportText": "Some report.",
        }
        r = client.post("/analyze-text", json=payload)
        assert r.status_code == 200

    def test_model_not_loaded_returns_503(self, client_unloaded: TestClient):
        r = client_unloaded.post("/analyze-text", json=FRAUD_PAYLOAD)
        assert r.status_code == 503


# ── /analyze-text — response contract ────────────────────────────────────────

class TestAnalyzeContract:
    def _call(self, client: TestClient, payload: dict = None) -> dict:
        r = client.post("/analyze-text", json=payload or FRAUD_PAYLOAD)
        assert r.status_code == 200, r.text
        return r.json()

    def test_required_fields_present(self, client: TestClient):
        body = self._call(client)
        for field in ("score", "label", "confidence", "claimScore",
                      "maintenanceScore", "combinedScore", "flaggedSignals"):
            assert field in body, f"Missing field: {field}"

    def test_score_in_range(self, client: TestClient):
        body = self._call(client)
        assert 0.0 <= body["score"] <= 100.0

    def test_label_is_valid(self, client: TestClient):
        body = self._call(client)
        assert body["label"] in ("FRAUD", "LEGITIMATE")

    def test_flagged_signals_is_list(self, client: TestClient):
        body = self._call(client)
        assert isinstance(body["flaggedSignals"], list)

    def test_component_scores_in_range(self, client: TestClient):
        body = self._call(client)
        for field in ("claimScore", "maintenanceScore", "combinedScore"):
            assert 0.0 <= body[field] <= 100.0, f"{field} out of range"

    def test_fraud_label_when_high_score(self, client: TestClient):
        """Mock returns 80 — final weighted score should be FRAUD."""
        body = self._call(client)
        assert body["label"] == "FRAUD"

    def test_legitimate_label_when_low_score(self, client_legitimate: TestClient):
        """Mock returns 10 — final weighted score should be LEGITIMATE."""
        body = self._call(client_legitimate, LEGITIMATE_PAYLOAD)
        assert body["label"] == "LEGITIMATE"


# ── /analyze-text — weighted score formula ───────────────────────────────────

class TestScoreFormula:
    def test_weighted_formula(self, client: TestClient):
        """
        Final score = 0.30×claim + 0.30×maintenance + 0.40×combined.
        Mock returns fixed score of 80 for all three calls.
        Expected: 0.30×80 + 0.30×80 + 0.40×80 = 80.0
        """
        r = client.post("/analyze-text", json=FRAUD_PAYLOAD)
        body = r.json()
        expected = round(
            body["claimScore"]       * 0.30 +
            body["maintenanceScore"] * 0.30 +
            body["combinedScore"]    * 0.40,
            1,
        )
        assert abs(body["score"] - expected) < 0.2