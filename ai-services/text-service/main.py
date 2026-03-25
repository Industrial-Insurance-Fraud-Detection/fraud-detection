from contextlib import asynccontextmanager
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import SERVICE_NAME, SERVICE_VERSION, PORT, FRAUD_THRESHOLD, SUSPICIOUS_PHRASES
from schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse


# ── lifespan: load model once at startup ─────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        from predict import load_model
        load_model()
        app.state.model_loaded = True
        print("NLP model loaded successfully")
    except Exception as e:
        print(f"Warning: NLP model could not be loaded: {e}")
        app.state.model_loaded = False
    yield


# ── app ───────────────────────────────────────────────────────

app = FastAPI(
    title=SERVICE_NAME,
    description="BERT multilingual fraud detection on claim descriptions and maintenance reports.",
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── routes ────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok" if app.state.model_loaded else "degraded",
        service=SERVICE_NAME,
        modelLoaded=app.state.model_loaded,
    )


@app.post("/analyze-text", response_model=AnalyzeResponse)
def analyze_text(request: AnalyzeRequest):
    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="NLP model not loaded")

    try:
        from predict import predict_fraud

        claim_result       = predict_fraud(request.claimDescription)
        maintenance_result = predict_fraud(request.maintenanceReportText)
        combined_result    = predict_fraud(
            f"CLAIM DESCRIPTION:\n{request.claimDescription}\n\n"
            f"MAINTENANCE REPORT:\n{request.maintenanceReportText}"
        )

        final_score = round(
            claim_result["fraud_score"]       * 0.30 +
            maintenance_result["fraud_score"] * 0.30 +
            combined_result["fraud_score"]    * 0.40,
            1,
        )

        flagged = _find_suspicious_phrases(
            request.claimDescription + " " + request.maintenanceReportText
        )

        return AnalyzeResponse(
            score=            final_score,
            label=            "FRAUD" if final_score >= FRAUD_THRESHOLD else "LEGITIMATE",
            confidence=       combined_result["confidence"],
            claimScore=       claim_result["fraud_score"],
            maintenanceScore= maintenance_result["fraud_score"],
            combinedScore=    combined_result["fraud_score"],
            flaggedSignals=   flagged,
        )

    except Exception as e:
        print(f"Error analyzing claim {request.claimId}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ── helpers ───────────────────────────────────────────────────

def _find_suspicious_phrases(text: str) -> list[str]:
    text_lower = text.lower()
    return [
        f"Suspicious phrase detected: '{phrase}'"
        for phrase in SUSPICIOUS_PHRASES
        if phrase.lower() in text_lower
    ]


# ── entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)