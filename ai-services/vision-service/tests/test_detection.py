"""
tests/test_detection.py

Unit tests for YOLOv8 detection scoring logic.
The model itself is always mocked — we test the scoring logic, not the weights.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from app.services.analyzer import _run_detection
from tests.conftest import make_mock_yolo


def _img() -> np.ndarray:
    """Dummy 200×200 BGR image for passing to _run_detection."""
    return np.full((200, 200, 3), 128, dtype=np.uint8)


class TestDetectionNoBoxes:
    def test_no_detections_scores_high(self):
        """Claim says damage exists but model finds nothing → fraud signal."""
        mock = make_mock_yolo(detections=None)
        with patch("app.services.analyzer._yolo_model", mock):
            score, boxes = _run_detection([_img()], declared_damage="scratches")
        assert score >= 70.0
        assert boxes == []

    def test_no_detections_without_declared_damage(self):
        mock = make_mock_yolo(detections=None)
        with patch("app.services.analyzer._yolo_model", mock):
            score, boxes = _run_detection([_img()], declared_damage=None)
        assert score >= 70.0


class TestDetectionMismatch:
    def test_wrong_class_scores_high(self):
        """Detected 'patches' but declared 'crazing' → mismatch."""
        mock = make_mock_yolo(detections=[
            {"class_id": 2, "confidence": 0.90,  # patches = index 2
             "x1": 10, "y1": 10, "x2": 100, "y2": 100}
        ])
        with patch("app.services.analyzer._yolo_model", mock):
            score, boxes = _run_detection([_img()], declared_damage="crazing")
        assert score >= 70.0
        assert len(boxes) == 1
        assert boxes[0].className == "patches"

    def test_mismatch_multiple_images(self):
        mock = make_mock_yolo(detections=[
            {"class_id": 4, "confidence": 0.85,   # rolled-in_scale
             "x1": 5, "y1": 5, "x2": 50, "y2": 50}
        ])
        with patch("app.services.analyzer._yolo_model", mock):
            score, _ = _run_detection([_img(), _img()], declared_damage="scratches")
        assert score >= 70.0


class TestDetectionMatch:
    def test_correct_class_scores_low(self):
        """Detected 'scratches' and declared 'scratches' → low score."""
        mock = make_mock_yolo(detections=[
            {"class_id": 5, "confidence": 0.92,  # scratches = index 5
             "x1": 10, "y1": 10, "x2": 80, "y2": 80}
        ])
        with patch("app.services.analyzer._yolo_model", mock):
            score, boxes = _run_detection([_img()], declared_damage="scratches")
        assert score < 35.0
        assert len(boxes) == 1

    def test_correct_class_no_declared_damage(self):
        """No declared damage provided — only boxes matter."""
        mock = make_mock_yolo(detections=[
            {"class_id": 0, "confidence": 0.75,  # crazing
             "x1": 0, "y1": 0, "x2": 50, "y2": 50}
        ])
        with patch("app.services.analyzer._yolo_model", mock):
            score, boxes = _run_detection([_img()], declared_damage=None)
        # declared=None → no mismatch possible → low score
        assert score < 35.0


class TestDetectionBoundingBoxFields:
    def test_box_has_correct_fields(self):
        mock = make_mock_yolo(detections=[
            {"class_id": 1, "confidence": 0.88,  # inclusion
             "x1": 10.5, "y1": 20.5, "x2": 90.5, "y2": 110.5}
        ])
        with patch("app.services.analyzer._yolo_model", mock):
            _, boxes = _run_detection([_img()], declared_damage=None)
        assert len(boxes) == 1
        b = boxes[0]
        assert b.className == "inclusion"
        assert b.imageIndex == 0
        assert 0 <= b.confidence <= 1.0
        assert b.x1 < b.x2
        assert b.y1 < b.y2

    def test_image_index_correct_for_second_image(self):
        """Box from the second image should have imageIndex=1."""
        # First image has no boxes, second has one
        import torch
        from unittest.mock import MagicMock

        mock_model = MagicMock()

        # Empty result for first image
        r0 = MagicMock()
        r0.boxes = None

        # Box for second image
        box = MagicMock()
        box.xyxy = [torch.tensor([5.0, 5.0, 50.0, 50.0])]
        box.cls = [torch.tensor(3)]   # pitted_surface
        box.conf = [torch.tensor(0.77)]
        boxes_mock = MagicMock()
        boxes_mock.__iter__ = MagicMock(return_value=iter([box]))
        r1 = MagicMock()
        r1.boxes = boxes_mock

        mock_model.predict.side_effect = [[r0], [r1]]

        with patch("app.services.analyzer._yolo_model", mock_model):
            _, boxes = _run_detection([_img(), _img()], declared_damage=None)

        box_from_second = [b for b in boxes if b.imageIndex == 1]
        assert len(box_from_second) == 1
        assert box_from_second[0].className == "pitted_surface"