"""
Pydantic schemas for vision-service request / response.

Request  : POST /analyze  ← called by the orchestrator worker
Response : VisionResponse ← returned to orchestrator for score aggregation
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


# ── Request ───────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """
    Body sent by the orchestrator to POST /analyze.

    photoPaths  : list of MinIO-signed URLs (http/https) or local paths (tests)
    declaredDamage : damage type declared by the claimant  e.g. "scratches"
    incidentDate   : ISO-8601 date string  e.g. "2026-02-10T00:00:00"
    claimId        : for traceability in logs
    """
    photoPaths: list[str] = Field(..., min_length=1)
    declaredDamage: Optional[str] = None
    incidentDate: Optional[datetime] = None
    claimId: Optional[str] = None

    @field_validator("photoPaths")
    @classmethod
    def paths_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("photoPaths must contain at least one path")
        return v


# ── Sub-models ────────────────────────────────────────────────────────────────

class BoundingBox(BaseModel):
    """Single detection result for one image."""
    imageIndex: int          # which photo in the input list (0-based)
    className: str           # e.g. "scratches"
    confidence: float        # 0.0 – 1.0
    x1: float
    y1: float
    x2: float
    y2: float


class ExifIssue(BaseModel):
    """One EXIF-based fraud indicator."""
    imageIndex: int
    issueType: str           # DATE_MISMATCH | EDITING_SOFTWARE | DEVICE_INCONSISTENCY
    detail: str              # human-readable description shown to investigator
    severity: str            # LOW | MEDIUM | HIGH | CRITICAL


# ── Response ──────────────────────────────────────────────────────────────────

class ComponentScores(BaseModel):
    detection: float = Field(..., ge=0, le=100)
    ela: float       = Field(..., ge=0, le=100)
    exif: float      = Field(..., ge=0, le=100)


class VisionResponse(BaseModel):
    """
    Standardised response consumed by the orchestrator.

    score        : final vision fraud score 0-100
    boxes        : all bounding-box detections across all photos
    manipulation : True if ANY photo's ELA score exceeds the threshold
    exifIssues   : list of EXIF anomalies found across all photos
    breakdown    : per-component scores for the investigator UI
    indicators   : string tags for the fraud report  e.g. ["DAMAGE_TYPE_MISMATCH"]
    processingMs : wall-clock time for observability
    """
    score: float              = Field(..., ge=0, le=100)
    boxes: list[BoundingBox]  = []
    manipulation: bool        = False
    exifIssues: list[ExifIssue] = []
    breakdown: ComponentScores
    indicators: list[str]     = []
    processingMs: int         = 0


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()} 
    status: str
    model_loaded: bool
    service: str