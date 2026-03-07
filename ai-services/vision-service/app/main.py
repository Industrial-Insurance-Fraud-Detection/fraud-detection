"""
vision-service  —  FastAPI application entry point.

Endpoints:
  GET  /health        liveness probe used by Docker + orchestrator
  POST /analyze       called by orchestrator worker (parallel with other services)
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import AnalyzeRequest, HealthResponse, VisionResponse
from app.services.analyzer import analyzer, get_model

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── lifespan: load model once at startup ─────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading YOLOv8 model at startup…")
    try:
        get_model()
        logger.info("YOLOv8 model ready ✓")
    except FileNotFoundError as exc:
        # Service starts anyway — /health will report model_loaded=False
        # This lets the orchestrator detect the issue via health checks
        logger.error("Model not found: %s", exc)
    yield
    logger.info("vision-service shutting down")


# ── app ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Taamine Vision Service",
    description=(
        "YOLOv8 damage detection + ELA manipulation analysis + EXIF forensics. "
        "Returns a vision fraud score 0-100 consumed by the AI orchestrator."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # internal service — orchestrator only
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """
    Liveness probe.
    Returns 200 with model_loaded=True when ready to serve /analyze.
    """
    from pathlib import Path
    model_loaded = Path(settings.MODEL_PATH).exists()
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        service=settings.SERVICE_NAME,
    )


@app.post("/analyze", response_model=VisionResponse, tags=["vision"])
async def analyze_images(request: AnalyzeRequest) -> VisionResponse:
    """
    Analyze one or more damage photos for fraud signals.

    Called by the orchestrator worker in parallel with the other 3 AI services.

    - **photoPaths**: list of MinIO signed URLs (or local paths in tests)
    - **declaredDamage**: damage type stated by the claimant
    - **incidentDate**: date of the claimed incident (ISO-8601)
    - **claimId**: for log traceability
    """
    try:
        result = analyzer.analyze(
            photo_paths=request.photoPaths,
            declared_damage=request.declaredDamage,
            incident_date=request.incidentDate,
            claim_id=request.claimId,
        )
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {exc}")
    except Exception as exc:
        logger.exception("Unhandled error in /analyze for claim %s", request.claimId)
        raise HTTPException(status_code=500, detail=str(exc))