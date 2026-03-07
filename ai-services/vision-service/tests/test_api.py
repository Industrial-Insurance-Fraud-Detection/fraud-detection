"""
tests/test_api.py

Tests for the HTTP layer:
  - GET  /health
  - POST /analyze — schema contract, status codes, edge cases
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import (
    make_clean_jpeg,
    make_manipulated_jpeg,
    make_mock_yolo,
    save_bytes_to_tmp,
)


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_returns_200(self, client: TestClient):
        r = client.get("/health")
        assert r.status_code == 200

    def test_response_schema(self, client: TestClient):
        body = client.get("/health").json()
        assert "status" in body
        assert "model_loaded" in body
        assert "service" in body

    def test_service_name(self, client: TestClient):
        body = client.get("/health").json()
        assert body["service"] == "vision-service"


# ── /analyze — contract ───────────────────────────────────────────────────────

class TestAnalyzeContract:
    """Verify the response always matches the VisionResponse schema."""

    def _post(self, client, paths: list[str], **kwargs) -> dict:
        payload = {"photoPaths": paths, **kwargs}
        r = client.post("/analyze", json=payload)
        assert r.status_code == 200, r.text
        return r.json()

    def test_required_fields_present(self, client: TestClient, clean_image_path: str):
        body = self._post(client, [clean_image_path])
        for field in ("score", "boxes", "manipulation", "exifIssues",
                      "breakdown", "indicators", "processingMs"):
            assert field in body, f"Missing field: {field}"

    def test_score_in_range(self, client: TestClient, clean_image_path: str):
        body = self._post(client, [clean_image_path])
        assert 0.0 <= body["score"] <= 100.0

    def test_breakdown_fields(self, client: TestClient, clean_image_path: str):
        body = self._post(client, [clean_image_path])
        bd = body["breakdown"]
        assert "detection" in bd
        assert "ela" in bd
        assert "exif" in bd
        for v in bd.values():
            assert 0.0 <= v <= 100.0

    def test_manipulation_is_bool(self, client: TestClient, clean_image_path: str):
        body = self._post(client, [clean_image_path])
        assert isinstance(body["manipulation"], bool)

    def test_processing_ms_positive(self, client: TestClient, clean_image_path: str):
        body = self._post(client, [clean_image_path])
        assert body["processingMs"] >= 0

    def test_multiple_photos_accepted(self, client: TestClient, tmp_path):
        paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.jpg"
            p.write_bytes(make_clean_jpeg(seed=i))
            paths.append(str(p))
        body = self._post(client, paths)
        assert 0 <= body["score"] <= 100


# ── /analyze — validation errors ─────────────────────────────────────────────

class TestAnalyzeValidation:
    def test_missing_photo_paths_422(self, client: TestClient):
        r = client.post("/analyze", json={})
        assert r.status_code == 422

    def test_empty_photo_paths_422(self, client: TestClient):
        r = client.post("/analyze", json={"photoPaths": []})
        assert r.status_code == 422

    def test_invalid_incident_date_422(self, client: TestClient, clean_image_path: str):
        r = client.post("/analyze", json={
            "photoPaths": [clean_image_path],
            "incidentDate": "not-a-date",
        })
        assert r.status_code == 422


# ── /analyze — optional fields ────────────────────────────────────────────────

class TestAnalyzeOptionalFields:
    def test_without_declared_damage(self, client: TestClient, clean_image_path: str):
        r = client.post("/analyze", json={"photoPaths": [clean_image_path]})
        assert r.status_code == 200

    def test_without_incident_date(self, client: TestClient, clean_image_path: str):
        r = client.post("/analyze", json={
            "photoPaths": [clean_image_path],
            "declaredDamage": "scratches",
        })
        assert r.status_code == 200

    def test_with_claim_id(self, client: TestClient, clean_image_path: str):
        r = client.post("/analyze", json={
            "photoPaths": [clean_image_path],
            "claimId": "SIN-2026-007823",
        })
        assert r.status_code == 200

    def test_bad_image_path_still_returns_200(self, client: TestClient):
        """Service should not crash on unreadable path — returns fallback score."""
        r = client.post("/analyze", json={
            "photoPaths": ["/nonexistent/path/photo.jpg"],
        })
        assert r.status_code == 200
        body = r.json()
        assert "IMAGE_LOAD_FAILED" in body["indicators"]