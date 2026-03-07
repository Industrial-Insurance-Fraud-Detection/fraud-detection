"""
Shared pytest fixtures for vision-service tests.

Key design decisions:
  - Tests NEVER hit MinIO or require a real YOLOv8 model.
  - The YOLOv8 model is mocked with a deterministic stub.
  - Real images are generated in-memory using numpy + PIL.
  - The ELA signature is written by the same helper used in production.
"""
from __future__ import annotations

import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ── image helpers ─────────────────────────────────────────────────────────────

def make_clean_jpeg(width: int = 200, height: int = 200, seed: int = 0) -> bytes:
    """
    Return JPEG bytes of a synthetic 'clean' industrial surface image.
    No ELA signature, no suspicious EXIF.
    """
    rng = np.random.default_rng(seed)
    base = np.full((height, width, 3), 128, dtype=np.uint8)
    noise = rng.integers(-15, 15, (height, width, 3))
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # add subtle scratch lines
    for y in [60, 80, 120, 140]:
        img[y:y+2, :] = np.clip(img[y:y+2, :].astype(np.int16) - 50, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(img, "RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def make_manipulated_jpeg(width: int = 200, height: int = 200, seed: int = 0) -> bytes:
    """
    Return JPEG bytes of a synthetic 'manipulated' image.
    ELA signature written: rows 0,2,4,…,14 set to pixel value 42.
    """
    data = make_clean_jpeg(width, height, seed)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    for r in range(0, 16, 2):
        img[r, :, :] = 42
    buf = io.BytesIO()
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return encoded.tobytes()


def make_jpeg_with_exif(
    software: str = "Canon EOS Digital",
    date_str: str = "2026:02:10 08:00:00",
    width: int = 200,
    height: int = 200,
) -> bytes:
    """Return JPEG bytes with embedded EXIF (software + date)."""
    import piexif

    data = make_clean_jpeg(width, height)
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Software: software.encode(),
            piexif.ImageIFD.DateTime: date_str.encode(),
        },
        "Exif": {
            piexif.ExifIFD.DateTimeOriginal: date_str.encode(),
            piexif.ExifIFD.DateTimeDigitized: date_str.encode(),
        },
        "GPS": {}, "1st": {}, "thumbnail": None,
    }
    exif_bytes = piexif.dump(exif_dict)
    # piexif.insert needs a file path — use temp file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(data)
        tmp_path = f.name
    piexif.insert(exif_bytes, tmp_path)
    result = Path(tmp_path).read_bytes()
    Path(tmp_path).unlink()
    return result


def save_bytes_to_tmp(data: bytes) -> str:
    """Write bytes to a temp JPEG file, return the path string."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(data)
        return f.name


# ── mock YOLOv8 model ─────────────────────────────────────────────────────────

def make_mock_yolo(detections: list[dict] | None = None):
    """
    Build a mock YOLO model that returns deterministic detections.

    detections: list of dicts with keys: class_id, confidence, x1, y1, x2, y2
    If None → empty detection (no boxes found).
    """
    import torch

    mock_model = MagicMock()

    if not detections:
        # no detections
        mock_result = MagicMock()
        mock_result.boxes = None
        mock_model.predict.return_value = [mock_result]
        return mock_model

    mock_boxes = MagicMock()
    box_list = []
    for d in detections:
        box = MagicMock()
        box.xyxy = [torch.tensor([d["x1"], d["y1"], d["x2"], d["y2"]])]
        box.cls = [torch.tensor(d["class_id"])]
        box.conf = [torch.tensor(d["confidence"])]
        box_list.append(box)

    mock_boxes.__iter__ = MagicMock(return_value=iter(box_list))
    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_model.predict.return_value = [mock_result]
    return mock_model


# ── FastAPI test client fixture ───────────────────────────────────────────────

@pytest.fixture()
def client():
    """
    TestClient with YOLOv8 loading mocked.
    The model stub returns no detections by default.
    Override _yolo_model in individual tests as needed.
    """
    from app import main as main_module
    import app.services.analyzer as analyzer_module

    with patch.object(analyzer_module, "_yolo_model", make_mock_yolo()):
        with patch.object(main_module, "get_model", return_value=make_mock_yolo()):
            from app.main import app
            with TestClient(app) as c:
                yield c


@pytest.fixture()
def clean_image_path(tmp_path) -> str:
    p = tmp_path / "clean.jpg"
    p.write_bytes(make_clean_jpeg())
    return str(p)


@pytest.fixture()
def manipulated_image_path(tmp_path) -> str:
    p = tmp_path / "manip.jpg"
    p.write_bytes(make_manipulated_jpeg())
    return str(p)