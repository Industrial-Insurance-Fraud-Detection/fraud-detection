"""
tests/test_ela.py

Unit tests for ELA manipulation detection.
No real model needed — pure image/numpy logic.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image
import numpy as np

from app.services.analyzer import _ela_score_single
from tests.conftest import make_clean_jpeg, make_manipulated_jpeg


class TestElaSingle:
    """_ela_score_single(bytes) → float 0-100"""

    def test_clean_image_scores_low(self):
        for seed in range(5):
            score = _ela_score_single(make_clean_jpeg(seed=seed))
            assert score < 25, f"Clean image scored {score} (seed={seed})"

    def test_manipulated_image_scores_high(self):
        for seed in range(5):
            score = _ela_score_single(make_manipulated_jpeg(seed=seed))
            assert score > 35, f"Manipulated image scored {score} (seed={seed})"

    def test_separation_is_large(self):
        """Manipulated score must be >> clean score — no overlap."""
        clean_scores = [_ela_score_single(make_clean_jpeg(seed=i)) for i in range(5)]
        manip_scores = [_ela_score_single(make_manipulated_jpeg(seed=i)) for i in range(5)]
        assert max(clean_scores) < min(manip_scores), (
            f"Scores overlap: clean max={max(clean_scores):.1f}, "
            f"manip min={min(manip_scores):.1f}"
        )

    def test_score_in_range(self):
        for make in (make_clean_jpeg, make_manipulated_jpeg):
            s = _ela_score_single(make())
            assert 0.0 <= s <= 100.0

    def test_invalid_bytes_returns_zero(self):
        score = _ela_score_single(b"this is not an image")
        assert score == 0.0

    def test_tiny_image_no_signature_rows(self):
        """
        Image with < 16 rows has no valid signature rows.
        Score is unpredictable (JPEG artifacts may hit any value),
        so we only assert it stays within valid range 0-100.
        """
        tiny = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
        buf = io.BytesIO()
        tiny.save(buf, format="JPEG")
        score = _ela_score_single(buf.getvalue())
        assert 0.0 <= score <= 100.0   # just check it doesn't crash

    def test_deterministic(self):
        """Same input always yields same score."""
        data = make_manipulated_jpeg(seed=99)
        s1 = _ela_score_single(data)
        s2 = _ela_score_single(data)
        assert s1 == s2