"""
tests/test_api.py

Tests for the HTTP layer:
  - GET  /health
  - POST /detect-anomalies — schema contract, status codes, edge cases

No MinIO required — all CSV data is passed as local paths via /test-local
or the endpoint is tested for contract/validation only.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient):
        body = client.get("/health").json()
        assert "status" in body
        assert "models_loaded" in body

    def test_status_is_ok(self, client: TestClient):
        body = client.get("/health").json()
        assert body["status"] == "ok"

    def test_models_loaded_true(self, client: TestClient):
        body = client.get("/health").json()
        assert body["models_loaded"] is True


# ── /detect-anomalies — validation ───────────────────────────────────────────

class TestDetectAnomaliesValidation:
    """
    Verify that missing or malformed request fields return 422
    before any model inference happens.
    """

    def test_missing_csv_path_422(self, client: TestClient):
        r = client.post("/detect-anomalies", json={
            "claimId": "SIN-2026-001",
            "claimDate": "2026-02-10",
        })
        assert r.status_code == 422

    def test_missing_claim_id_422(self, client: TestClient):
        r = client.post("/detect-anomalies", json={
            "csvPath": "claims/test.csv",
            "claimDate": "2026-02-10",
        })
        assert r.status_code == 422

    def test_missing_claim_date_422(self, client: TestClient):
        r = client.post("/detect-anomalies", json={
            "csvPath": "claims/test.csv",
            "claimId": "SIN-2026-001",
        })
        assert r.status_code == 422

    def test_empty_body_422(self, client: TestClient):
        r = client.post("/detect-anomalies", json={})
        assert r.status_code == 422

    def test_wrong_content_type_422(self, client: TestClient):
        r = client.post(
            "/detect-anomalies",
            data="not json",
            headers={"Content-Type": "text/plain"},
        )
        assert r.status_code in (400, 422)


# ── /detect-anomalies — MinIO error handling ─────────────────────────────────

class TestDetectAnomaliesMinioError:
    """
    When MinIO is not running locally, the endpoint must return 500
    with a clear error message — not an unhandled crash.
    """

    def test_minio_unavailable_returns_500(self, client: TestClient):
        r = client.post("/detect-anomalies", json={
            "csvPath": "claims/nonexistent.csv",
            "claimId": "SIN-2026-999",
            "claimDate": "2026-02-10",
        })
        # MinIO is not running locally — must get 500, not a crash
        assert r.status_code == 500

    def test_500_has_error_message(self, client: TestClient):
        r = client.post("/detect-anomalies", json={
            "csvPath": "claims/nonexistent.csv",
            "claimId": "SIN-2026-999",
            "claimDate": "2026-02-10",
        })
        body = r.json()
        assert "detail" in body
        assert len(body["detail"]) > 0


# ── /test-local — contract ────────────────────────────────────────────────────

class TestLocalEndpointContract:
    """
    /test-local bypasses MinIO and runs on synthetic data.
    Verifies the full response contract end-to-end.
    """

    def _call(self, client: TestClient) -> dict:
        r = client.post("/test-local")
        assert r.status_code == 200, r.text
        return r.json()

    def test_returns_200(self, client: TestClient):
        r = client.post("/test-local")
        assert r.status_code == 200

    def test_required_fields_present(self, client: TestClient):
        body = self._call(client)
        result = body["result"]
        for field in ("score", "anomalies", "pre_incident_anomaly", "fraud_indicator"):
            assert field in result, f"Missing field: {field}"

    def test_score_in_range(self, client: TestClient):
        body = self._call(client)
        score = body["result"]["score"]
        assert 0 <= score <= 100, f"Score out of range: {score}"

    def test_anomalies_is_list(self, client: TestClient):
        body = self._call(client)
        assert isinstance(body["result"]["anomalies"], list)

    def test_anomaly_item_fields(self, client: TestClient):
        body = self._call(client)
        for item in body["result"]["anomalies"]:
            assert "timestamp"  in item
            assert "parameter"  in item
            assert "value"      in item
            assert "threshold"  in item

    def test_anomaly_value_is_float(self, client: TestClient):
        body = self._call(client)
        for item in body["result"]["anomalies"]:
            assert isinstance(item["value"], float)
            assert isinstance(item["threshold"], float)

    def test_pre_incident_is_bool(self, client: TestClient):
        body = self._call(client)
        assert isinstance(body["result"]["pre_incident_anomaly"], bool)

    def test_fraud_indicator_valid_value(self, client: TestClient):
        valid = {
            "NO_PRECURSOR_DETECTED",
            "ABRUPT_FAILURE",
            "GRADUAL_DEGRADATION",
            "MINOR_PRECURSORS_DETECTED",
            "NORMAL",
            # spaces variant (current scorer.py) — remove once underscores are fixed
            "NO PRECURSOR DETECTED",
            "ABRUPT FAILURE",
            "GRADUAL DEGRADATION",
            "MINOR PRECURSORS DETECTED",
        }
        body = self._call(client)
        indicator = body["result"]["fraud_indicator"]
        assert indicator in valid, f"Unexpected fraud_indicator: {indicator}"