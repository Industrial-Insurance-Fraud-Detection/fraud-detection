"""
tests/test_ela.py

Unit tests for real ELA (Error Level Analysis) manipulation detection.
No YOLOv8 model needed — pure image/numpy logic.

Real ELA works by re-compressing at quality=75 and measuring pixel
difference. Manipulated regions show higher error levels than authentic ones.
"""
from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw
import numpy as np

from app.services.analyzer import _ela_score_single


def _make_authentic_jpeg(seed: int = 0) -> bytes:
    """
    Authentic JPEG — compressed once from a natural-looking image.
    Re-compressing it again changes very little → low ELA score.
    """
    rng = np.random.default_rng(seed)
    # gradient + noise mimics a real photo better than flat colour
    base = np.tile(np.linspace(80, 180, 200, dtype=np.uint8), (200, 1))
    noise = rng.integers(-20, 20, (200, 200)).astype(np.int16)
    arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(np.stack([arr, arr, arr], axis=2))
    buf = io.BytesIO()
    # save at quality=95 first (simulates original camera JPEG)
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _make_manipulated_jpeg(seed: int = 0) -> bytes:
    """
    Manipulated JPEG — a region was pasted from a differently-compressed source.
    This creates a visible ELA discontinuity in that region.
    """
    # base image
    base_bytes = _make_authentic_jpeg(seed)
    img = Image.open(io.BytesIO(base_bytes)).convert("RGB")

    # create a patch saved at different quality (simulates copy-paste)
    rng = np.random.default_rng(seed + 100)
    patch_arr = rng.integers(50, 200, (60, 60, 3), dtype=np.uint8)
    patch_img = Image.fromarray(patch_arr)

    # compress patch at different quality → different error level signature
    patch_buf = io.BytesIO()
    patch_img.save(patch_buf, format="JPEG", quality=30)
    patch_buf.seek(0)
    patch_reloaded = Image.open(patch_buf).convert("RGB")

    # paste manipulated patch into original
    img.paste(patch_reloaded, (70, 70))

    # save final composite
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=95)
    return out.getvalue()


class TestElaSingle:
    """_ela_score_single(bytes) → float 0-100"""

    def test_score_in_range(self):
        """Score must always be 0-100 regardless of input."""
        for make in (_make_authentic_jpeg, _make_manipulated_jpeg):
            s = _ela_score_single(make())
            assert 0.0 <= s <= 100.0, f"Score out of range: {s}"

    def test_invalid_bytes_returns_zero(self):
        assert _ela_score_single(b"not an image") == 0.0

    def test_tiny_image_returns_zero(self):
        tiny = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
        buf = io.BytesIO()
        tiny.save(buf, format="JPEG")
        assert _ela_score_single(buf.getvalue()) == 0.0

    def test_deterministic(self):
        """Same bytes → same score every time."""
        data = _make_manipulated_jpeg(seed=42)
        assert _ela_score_single(data) == _ela_score_single(data)

    def test_manipulated_scores_higher_than_authentic(self):
        """
        Core property: manipulated images must score higher than authentic ones
        on average across multiple seeds.
        """
        auth_scores  = [_ela_score_single(_make_authentic_jpeg(seed=i))   for i in range(8)]
        manip_scores = [_ela_score_single(_make_manipulated_jpeg(seed=i)) for i in range(8)]
        assert np.mean(manip_scores) > np.mean(auth_scores), (
            f"Expected manipulated > authentic on average.\n"
            f"  Authentic  mean={np.mean(auth_scores):.1f}  scores={[round(s,1) for s in auth_scores]}\n"
            f"  Manipulated mean={np.mean(manip_scores):.1f} scores={[round(s,1) for s in manip_scores]}"
        )

    def test_heavily_manipulated_scores_above_threshold(self):
        """
        Heavily copy-pasted image (large patch, extreme quality difference)
        should exceed the fraud threshold.
        """
        # build image with very aggressive manipulation
        base = Image.fromarray(
            np.full((300, 300, 3), 128, dtype=np.uint8)
        )
        # paste a large region saved at quality=5 (extreme compression)
        patch = Image.fromarray(
            np.random.randint(0, 255, (150, 150, 3), dtype=np.uint8)
        )
        buf = io.BytesIO()
        patch.save(buf, format="JPEG", quality=5)
        buf.seek(0)
        patch = Image.open(buf).convert("RGB")
        base.paste(patch, (50, 50))
        out = io.BytesIO()
        base.save(out, format="JPEG", quality=95)

        score = _ela_score_single(out.getvalue())
        assert score > 20.0, f"Heavy manipulation scored only {score}"