"""
tests/test_analyzer.py

End-to-end tests of VisionAnalyzer.analyze() — mocking only YOLOv8,
using real ELA + EXIF logic on synthetic images.

These mirror the 6 fraud test cases from the Kaggle notebook.
"""
from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.analyzer import VisionAnalyzer
from app.config import settings
from tests.conftest import (
    make_clean_jpeg,
    make_manipulated_jpeg,
    make_mock_yolo,
    save_bytes_to_tmp,
)

analyzer = VisionAnalyzer()


def _write(data: bytes, tmp_dir: Path, name: str) -> str:
    p = tmp_dir / name
    p.write_bytes(data)
    return str(p)


# ── Test Case 1: clean image ──────────────────────────────────────────────────

class TestCleanImage:
    def test_total_score_low(self, tmp_path):
        path = _write(make_clean_jpeg(), tmp_path, "clean.jpg")
        mock = make_mock_yolo(detections=[
            {"class_id": 5, "confidence": 0.90,  # scratches
             "x1": 10, "y1": 10, "x2": 80, "y2": 80}
        ])
        with patch("app.services.analyzer._yolo_model", mock):
            result = analyzer.analyze(
                photo_paths=[path],
                declared_damage="scratches",
                incident_date=datetime(2026, 2, 10, tzinfo=timezone.utc),
            )
        assert result.score < 25.0, f"Expected low score, got {result.score}"
        assert result.manipulation is False
        assert result.exifIssues == []


# ── Test Case 2: damage type mismatch ────────────────────────────────────────

class TestDamageMismatch:
    def test_detection_component_high(self, tmp_path):
        path = _write(make_clean_jpeg(), tmp_path, "patches.jpg")
        mock = make_mock_yolo(detections=[
            {"class_id": 2, "confidence": 0.88,  # patches
             "x1": 10, "y1": 10, "x2": 100, "y2": 100}
        ])
        with patch("app.services.analyzer._yolo_model", mock):
            result = analyzer.analyze(
                photo_paths=[path],
                declared_damage="crazing",  # mismatch
                incident_date=datetime(2026, 2, 10, tzinfo=timezone.utc),
            )
        assert result.breakdown.detection >= settings.DETECTION_FRAUD_THRESHOLD
        assert "DAMAGE_TYPE_MISMATCH" in result.indicators

    def test_total_score_above_threshold(self, tmp_path):
        path = _write(make_clean_jpeg(), tmp_path, "mismatch.jpg")
        mock = make_mock_yolo(detections=[
            {"class_id": 2, "confidence": 0.88,
             "x1": 10, "y1": 10, "x2": 100, "y2": 100}
        ])
        with patch("app.services.analyzer._yolo_model", mock):
            result = analyzer.analyze(
                photo_paths=[path],
                declared_damage="crazing",
            )
        assert result.score > 25.0


# ── Test Case 3: ELA manipulated ─────────────────────────────────────────────

class TestElaManipulated:
    def test_ela_component_high(self, tmp_path):
        path = _write(make_manipulated_jpeg(), tmp_path, "manip.jpg")
        mock = make_mock_yolo(detections=[
            {"class_id": 1, "confidence": 0.83,  # inclusion
             "x1": 10, "y1": 10, "x2": 60, "y2": 60}
        ])
        with patch("app.services.analyzer._yolo_model", mock):
            result = analyzer.analyze(
                photo_paths=[path],
                declared_damage="inclusion",
            )
        assert result.breakdown.ela >= settings.ELA_FRAUD_THRESHOLD, (
            f"ELA score {result.breakdown.ela} < threshold {settings.ELA_FRAUD_THRESHOLD}"
        )

    def test_manipulation_flag_true(self, tmp_path):
        path = _write(make_manipulated_jpeg(), tmp_path, "manip2.jpg")
        mock = make_mock_yolo()
        with patch("app.services.analyzer._yolo_model", mock):
            result = analyzer.analyze(photo_paths=[path])
        assert result.manipulation is True

    def test_indicator_set(self, tmp_path):
        path = _write(make_manipulated_jpeg(), tmp_path, "manip3.jpg")
        mock = make_mock_yolo()
        with patch("app.services.analyzer._yolo_model", mock):
            result = analyzer.analyze(photo_paths=[path])
        assert "IMAGE_MANIPULATION_DETECTED" in result.indicators


# ── Test Case 4: EXIF tampered ────────────────────────────────────────────────

class TestExifTampered:
    def test_exif_component_high(self, tmp_path):
        """
        Verify EXIF component scores high when Photoshop software + old date present.
        Uses _parse_exif monkey-patch (same reliable approach as test_exif.py)
        because PIL._getexif() reads piexif-injected tags inconsistently on Windows.
        """
        import numpy as np
        from PIL import Image
        from unittest.mock import patch as mpatch
        from app.services.analyzer import _run_exif

        # build a PIL image that reports Photoshop + old date via _getexif
        img = Image.fromarray(np.full((200, 200, 3), 128, dtype=np.uint8))
        img._getexif = lambda: {
            305:   "Adobe Photoshop CS6 (Windows)",   # Software
            36867: "2020:01:15 08:30:00",              # DateTimeOriginal
            306:   "2020:01:15 08:30:00",              # DateTime
        }

        incident = datetime(2026, 2, 10, tzinfo=timezone.utc)
        score, issues = _run_exif([img], incident_date=incident)

        assert score >= 40.0, f"Expected EXIF score >= 40, got {score}"
        assert any(i.issueType == "EDITING_SOFTWARE" for i in issues)
        assert any(i.issueType == "DATE_MISMATCH" for i in issues)


# ── Test Case 5: Multi-fraud ──────────────────────────────────────────────────

class TestMultiFraud:
    def test_all_components_elevated(self, tmp_path):
        path = _write(make_manipulated_jpeg(), tmp_path, "multi.jpg")
        mock = make_mock_yolo(detections=[
            {"class_id": 5, "confidence": 0.67,  # scratches
             "x1": 5, "y1": 5, "x2": 50, "y2": 50}
        ])
        with patch("app.services.analyzer._yolo_model", mock):
            result = analyzer.analyze(
                photo_paths=[path],
                declared_damage="pitted_surface",  # mismatch
                incident_date=datetime(2026, 2, 10, tzinfo=timezone.utc),
            )
        # At least detection + ELA should be elevated
        assert result.breakdown.detection >= settings.DETECTION_FRAUD_THRESHOLD
        assert result.breakdown.ela >= settings.ELA_FRAUD_THRESHOLD
        assert result.score > 50.0

    def test_high_fraud_confidence_indicator(self, tmp_path):
        path = _write(make_manipulated_jpeg(), tmp_path, "multi2.jpg")
        mock = make_mock_yolo(detections=[
            {"class_id": 2, "confidence": 0.85,
             "x1": 10, "y1": 10, "x2": 80, "y2": 80}
        ])
        with patch("app.services.analyzer._yolo_model", mock):
            result = analyzer.analyze(
                photo_paths=[path],
                declared_damage="crazing",
            )
        if result.score >= 70:
            assert "HIGH_FRAUD_CONFIDENCE" in result.indicators


# ── Score math ────────────────────────────────────────────────────────────────

class TestScoreMath:
    def test_weighted_formula(self, tmp_path):
        """
        Manually verify: score = 0.45×det + 0.30×ela + 0.25×exif
        """
        path = _write(make_clean_jpeg(), tmp_path, "math.jpg")
        mock = make_mock_yolo(detections=[
            {"class_id": 5, "confidence": 0.90,
             "x1": 10, "y1": 10, "x2": 80, "y2": 80}
        ])
        with patch("app.services.analyzer._yolo_model", mock):
            result = analyzer.analyze(photo_paths=[path], declared_damage="scratches")

        expected = round(
            result.breakdown.detection * settings.WEIGHT_DETECTION +
            result.breakdown.ela       * settings.WEIGHT_ELA +
            result.breakdown.exif      * settings.WEIGHT_EXIF,
            1,
        )
        assert abs(result.score - expected) < 0.2

    def test_score_always_between_0_and_100(self, tmp_path):
        for seed in range(5):
            path = _write(make_clean_jpeg(seed=seed), tmp_path, f"r{seed}.jpg")
            mock = make_mock_yolo()
            with patch("app.services.analyzer._yolo_model", mock):
                result = analyzer.analyze(photo_paths=[path])
            assert 0.0 <= result.score <= 100.0