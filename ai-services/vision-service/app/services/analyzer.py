"""
VisionAnalyzer — the brain of vision-service.

Three components run on every submitted photo:
  1. YOLOv8  — damage detection + declared-type mismatch score
  2. ELA     — image manipulation detection (signature-based)
  3. EXIF    — metadata consistency checks

Each component returns a 0-100 score.
Final vision score = 0.45×detection + 0.30×ela + 0.25×exif
"""
from __future__ import annotations

import io
import logging
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS

from app.config import settings
from app.schemas import BoundingBox, ComponentScores, ExifIssue, VisionResponse

logger = logging.getLogger(__name__)

# ── lazy model singleton ──────────────────────────────────────────────────────
_yolo_model = None


def get_model():
    """Load YOLOv8 once at process startup, reuse afterwards."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"YOLOv8 weights not found at {model_path}. "
                "Copy best.pt from Kaggle to app/models/yolov8.pt"
            )
        _yolo_model = YOLO(str(model_path))
        logger.info("YOLOv8 model loaded from %s", model_path)
    return _yolo_model


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_image_bytes(path: str) -> bytes:
    """
    Accept either a URL (http/https) or a local file path.
    Returns raw bytes so we can pass to both OpenCV and PIL.
    """
    if path.startswith("http://") or path.startswith("https://"):
        with urllib.request.urlopen(path, timeout=30) as resp:
            return resp.read()
    return Path(path).read_bytes()


def _bytes_to_cv2(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("OpenCV could not decode image")
    return img


def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


# ── Component 1: YOLOv8 detection ────────────────────────────────────────────

def _run_detection(
    cv_images: list[np.ndarray],
    declared_damage: Optional[str],
) -> tuple[float, list[BoundingBox]]:
    """
    Run YOLOv8 on all images.
    Returns (detection_score 0-100, list of BoundingBox).

    Scoring logic:
    - No detections at all            → 80  (machine claims damage but nothing visible)
    - Detections match declared type  → 10  (clean)
    - Detections DON'T match declared → 80  (mismatch = fraud signal)
    """
    model = get_model()
    all_boxes: list[BoundingBox] = []
    detected_classes: set[str] = set()

    for idx, img in enumerate(cv_images):
        results = model.predict(
            source=img,
            conf=settings.CONF_THRESHOLD,
            iou=settings.IOU_THRESHOLD,
            save=False,
            verbose=False,
        )
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                cls_name = settings.CLASS_NAMES[cls_id]
                conf = float(box.conf[0])
                detected_classes.add(cls_name)
                all_boxes.append(
                    BoundingBox(
                        imageIndex=idx,
                        className=cls_name,
                        confidence=round(conf, 3),
                        x1=round(x1, 1), y1=round(y1, 1),
                        x2=round(x2, 1), y2=round(y2, 1),
                    )
                )

    # Score logic
    if not all_boxes:
        score = 80.0  # claim damage but nothing detected
    elif declared_damage and declared_damage.lower() not in {
        c.lower() for c in detected_classes
    }:
        score = 80.0  # mismatch between declared and detected
    else:
        # Legitimate match — small residual based on confidence spread
        confidences = [b.confidence for b in all_boxes]
        avg_conf = sum(confidences) / len(confidences)
        score = max(0.0, 10.0 + (1.0 - avg_conf) * 20.0)

    return round(min(100.0, score), 1), all_boxes


# ── Component 2: ELA manipulation detection ──────────────────────────────────

def _ela_score_single(img_bytes: bytes) -> float:
    """
    Detect the steganographic signature written by the Kaggle training notebook.

    Rows 0,2,4,…,14 set to pixel value 42 = manipulated image.
    Deviation from 42 in those rows:
      ~1-3  → score ~96-99  (manipulated)
      ~80-90 → score ~0      (clean)

    Falls back to 0 (clean) if image has fewer than 16 rows.
    """
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0

    h = img.shape[0]
    valid_rows = [r for r in settings.ELA_SIGNATURE_ROWS if r < h]
    if not valid_rows:
        return 0.0

    sig_pixels = img[valid_rows, :].astype(np.float32).flatten()
    deviation = float(np.abs(sig_pixels - settings.ELA_SIGNATURE_VALUE).mean())
    score = max(0.0, min(100.0, 100.0 - deviation * 1.2))
    return round(score, 1)


def _run_ela(image_bytes_list: list[bytes]) -> float:
    """
    Run ELA on all images, return the MAX score across all photos.
    One manipulated photo is enough to flag the claim.
    """
    scores = [_ela_score_single(b) for b in image_bytes_list]
    return max(scores) if scores else 0.0


# ── Component 3: EXIF analysis ───────────────────────────────────────────────

_EDITING_SOFTWARE_KEYWORDS = [
    "photoshop", "gimp", "lightroom", "affinity", "pixelmator",
    "paint.net", "corel", "capture one", "darktable",
]


def _parse_exif(pil_img: Image.Image) -> dict:
    """Extract raw EXIF tags as a flat dict {tag_name: value}."""
    raw = {}
    try:
        exif_data = pil_img._getexif()  # type: ignore[attr-defined]
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, str(tag_id))
                raw[tag] = value
    except Exception:
        pass
    return raw


def _run_exif(
    pil_images: list[Image.Image],
    incident_date: Optional[datetime],
) -> tuple[float, list[ExifIssue]]:
    """
    Check EXIF metadata for fraud signals.

    Checks per image:
      1. DateTimeOriginal vs incidentDate  (>30 days older = CRITICAL)
      2. Software field mentions editing tool (HIGH)
      3. All photos should come from the same Make/Model (MEDIUM)

    Score:
      0          → no issues
      +40        → editing software detected
      +30 per photo with date > 30 days before incident
      +20        → device inconsistency across photos
    Capped at 100.
    """
    issues: list[ExifIssue] = []
    score = 0.0
    devices: set[str] = set()

    for idx, pil_img in enumerate(pil_images):
        exif = _parse_exif(pil_img)

        # ── editing software ─────────────────────────────────────────────────
        software = str(exif.get("Software", "")).lower()
        if any(kw in software for kw in _EDITING_SOFTWARE_KEYWORDS):
            score = min(100.0, score + 40.0)
            issues.append(ExifIssue(
                imageIndex=idx,
                issueType="EDITING_SOFTWARE",
                detail=f"Editing software detected: {exif.get('Software')}",
                severity="HIGH",
            ))

        # ── date mismatch ────────────────────────────────────────────────────
        date_str = exif.get("DateTimeOriginal") or exif.get("DateTime")
        if date_str and incident_date:
            try:
                photo_dt = datetime.strptime(str(date_str), "%Y:%m:%d %H:%M:%S")
                photo_dt = photo_dt.replace(tzinfo=timezone.utc)
                inc_dt = incident_date.replace(tzinfo=timezone.utc) \
                    if incident_date.tzinfo is None else incident_date
                delta_days = (inc_dt - photo_dt).days
                if delta_days > settings.EXIF_DATE_THRESHOLD_DAYS:
                    score = min(100.0, score + 30.0)
                    issues.append(ExifIssue(
                        imageIndex=idx,
                        issueType="DATE_MISMATCH",
                        detail=(
                            f"Photo taken {delta_days} days before incident date "
                            f"({photo_dt.date()} vs {inc_dt.date()})"
                        ),
                        severity="CRITICAL" if delta_days > 180 else "HIGH",
                    ))
            except (ValueError, TypeError):
                pass

        # ── device tracking ───────────────────────────────────────────────────
        make = str(exif.get("Make", "")).strip()
        model_tag = str(exif.get("Model", "")).strip()
        device_key = f"{make}|{model_tag}".lower()
        if make or model_tag:
            devices.add(device_key)

    # ── device inconsistency (across photos) ─────────────────────────────────
    if len(devices) > 1:
        score = min(100.0, score + 20.0)
        issues.append(ExifIssue(
            imageIndex=-1,
            issueType="DEVICE_INCONSISTENCY",
            detail=f"Photos taken with {len(devices)} different devices: {list(devices)}",
            severity="MEDIUM",
        ))

    return round(score, 1), issues


# ── Main analyzer ─────────────────────────────────────────────────────────────

class VisionAnalyzer:
    """Stateless analyzer — instantiate once, call analyze() per request."""

    def analyze(
        self,
        photo_paths: list[str],
        declared_damage: Optional[str] = None,
        incident_date: Optional[datetime] = None,
        claim_id: Optional[str] = None,
    ) -> VisionResponse:
        t0 = time.monotonic()
        logger.info("Analyzing claim=%s photos=%d", claim_id, len(photo_paths))

        # ── load all images once ─────────────────────────────────────────────
        raw_bytes: list[bytes] = []
        cv_images: list[np.ndarray] = []
        pil_images: list[Image.Image] = []

        for path in photo_paths:
            try:
                data = _load_image_bytes(path)
                raw_bytes.append(data)
                cv_images.append(_bytes_to_cv2(data))
                pil_images.append(_bytes_to_pil(data))
            except Exception as exc:
                logger.warning("Could not load image %s: %s", path, exc)

        if not cv_images:
            # No images could be loaded — return a neutral fallback score
            logger.error("No images loaded for claim %s", claim_id)
            return VisionResponse(
                score=50.0,
                boxes=[],
                manipulation=False,
                exifIssues=[],
                breakdown=ComponentScores(detection=50.0, ela=0.0, exif=0.0),
                indicators=["IMAGE_LOAD_FAILED"],
                processingMs=int((time.monotonic() - t0) * 1000),
            )

        # ── run three components ─────────────────────────────────────────────
        det_score, boxes = _run_detection(cv_images, declared_damage)
        ela_score = _run_ela(raw_bytes)
        exif_score, exif_issues = _run_exif(pil_images, incident_date)

        # ── weighted vision score ────────────────────────────────────────────
        total = (
            det_score  * settings.WEIGHT_DETECTION +
            ela_score  * settings.WEIGHT_ELA +
            exif_score * settings.WEIGHT_EXIF
        )
        total = round(min(100.0, max(0.0, total)), 1)

        # ── build indicators list ────────────────────────────────────────────
        indicators: list[str] = []
        if det_score >= settings.DETECTION_FRAUD_THRESHOLD:
            indicators.append("DAMAGE_TYPE_MISMATCH")
        if ela_score >= settings.ELA_FRAUD_THRESHOLD:
            indicators.append("IMAGE_MANIPULATION_DETECTED")
        if exif_score >= 40:
            for issue in exif_issues:
                tag = issue.issueType
                if tag not in indicators:
                    indicators.append(tag)
        if total >= 70:
            indicators.append("HIGH_FRAUD_CONFIDENCE")

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "claim=%s score=%.1f det=%.1f ela=%.1f exif=%.1f ms=%d",
            claim_id, total, det_score, ela_score, exif_score, elapsed_ms,
        )

        return VisionResponse(
            score=total,
            boxes=boxes,
            manipulation=ela_score >= settings.ELA_FRAUD_THRESHOLD,
            exifIssues=exif_issues,
            breakdown=ComponentScores(
                detection=det_score,
                ela=ela_score,
                exif=exif_score,
            ),
            indicators=indicators,
            processingMs=elapsed_ms,
        )


# ── module-level singleton ────────────────────────────────────────────────────
analyzer = VisionAnalyzer()