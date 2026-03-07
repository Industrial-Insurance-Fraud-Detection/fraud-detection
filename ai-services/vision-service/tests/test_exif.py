"""
tests/test_exif.py

Unit tests for EXIF metadata fraud detection.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from PIL import Image
import io
import numpy as np

from app.services.analyzer import _run_exif, _parse_exif


def _blank_pil(width: int = 100, height: int = 100) -> Image.Image:
    return Image.fromarray(
        np.full((height, width, 3), 128, dtype=np.uint8), "RGB"
    )


class TestExifClean:
    def test_no_exif_no_issues(self):
        imgs = [_blank_pil()]
        score, issues = _run_exif(imgs, incident_date=None)
        assert score == 0.0
        assert issues == []

    def test_score_zero_when_clean(self):
        score, _ = _run_exif([_blank_pil()], incident_date=None)
        assert score == 0.0


class TestExifEditingSoftware:
    def _make_img_with_software(self, software: str) -> Image.Image:
        """
        Build a PIL image with Software EXIF set.
        We monkey-patch _getexif since PIL Image objects don't expose it easily
        without a real JPEG.
        """
        img = _blank_pil()
        img._getexif = lambda: {
            305: software,  # tag 305 = Software
        }
        return img

    def test_photoshop_raises_score(self):
        img = self._make_img_with_software("Adobe Photoshop CS6 (Windows)")
        score, issues = _run_exif([img], incident_date=None)
        assert score >= 40.0
        assert any(i.issueType == "EDITING_SOFTWARE" for i in issues)

    def test_gimp_raises_score(self):
        img = self._make_img_with_software("GIMP 2.10")
        score, issues = _run_exif([img], incident_date=None)
        assert score >= 40.0

    def test_canonical_camera_no_issue(self):
        img = self._make_img_with_software("Canon EOS Digital Solution Disk")
        score, issues = _run_exif([img], incident_date=None)
        assert not any(i.issueType == "EDITING_SOFTWARE" for i in issues)


class TestExifDateMismatch:
    def _img_with_date(self, date_str: str) -> Image.Image:
        img = _blank_pil()
        img._getexif = lambda: {
            306: date_str,   # DateTime
            36867: date_str, # DateTimeOriginal
        }
        return img

    def test_old_photo_raises_score(self):
        incident = datetime(2026, 2, 10, tzinfo=timezone.utc)
        old_photo = self._img_with_date("2025:06:01 12:00:00")  # ~250 days before
        score, issues = _run_exif([old_photo], incident_date=incident)
        assert score >= 30.0
        assert any(i.issueType == "DATE_MISMATCH" for i in issues)

    def test_recent_photo_no_issue(self):
        incident = datetime(2026, 2, 10, tzinfo=timezone.utc)
        recent = self._img_with_date("2026:02:09 10:00:00")  # 1 day before
        score, issues = _run_exif([recent], incident_date=incident)
        assert not any(i.issueType == "DATE_MISMATCH" for i in issues)

    def test_very_old_photo_critical_severity(self):
        incident = datetime(2026, 2, 10, tzinfo=timezone.utc)
        very_old = self._img_with_date("2023:01:01 00:00:00")  # >180 days
        _, issues = _run_exif([very_old], incident_date=incident)
        date_issues = [i for i in issues if i.issueType == "DATE_MISMATCH"]
        assert date_issues
        assert date_issues[0].severity == "CRITICAL"

    def test_no_incident_date_skips_check(self):
        img = self._img_with_date("2020:01:01 00:00:00")
        score, issues = _run_exif([img], incident_date=None)
        assert not any(i.issueType == "DATE_MISMATCH" for i in issues)


class TestExifDeviceInconsistency:
    def _img_with_device(self, make: str, model: str) -> Image.Image:
        img = _blank_pil()
        img._getexif = lambda: {
            271: make,   # Make
            272: model,  # Model
        }
        return img

    def test_different_devices_raises_score(self):
        imgs = [
            self._img_with_device("Canon", "EOS 5D"),
            self._img_with_device("Samsung", "Galaxy S23"),
        ]
        score, issues = _run_exif(imgs, incident_date=None)
        assert score >= 20.0
        assert any(i.issueType == "DEVICE_INCONSISTENCY" for i in issues)

    def test_same_device_no_issue(self):
        imgs = [
            self._img_with_device("Canon", "EOS 5D"),
            self._img_with_device("Canon", "EOS 5D"),
        ]
        _, issues = _run_exif(imgs, incident_date=None)
        assert not any(i.issueType == "DEVICE_INCONSISTENCY" for i in issues)


class TestExifScoreCap:
    def test_score_never_exceeds_100(self):
        """Pile up every issue on one image — score must stay ≤ 100."""
        incident = datetime(2026, 2, 10, tzinfo=timezone.utc)

        def make_img():
            img = _blank_pil()
            img._getexif = lambda: {
                305: "Adobe Photoshop CC",
                36867: "2020:01:01 00:00:00",  # 6+ years old
                271: f"Brand_{id(img)}",        # unique brand each call → inconsistency
            }
            return img

        imgs = [make_img(), make_img()]
        score, _ = _run_exif(imgs, incident_date=incident)
        assert score <= 100.0