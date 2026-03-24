"""
main.py
-------
This is the FastAPI "waiter".

It creates a mini web server with ONE main endpoint:
  POST /analyze-text

Your teammate's NestJS backend sends text here.
This file gives it to predict.py, then sends the score back.

Think of it like a drive-through window:
  - Car arrives (NestJS sends text)
  - Window takes the order (FastAPI receives it)
  - Kitchen makes the food (predict.py runs the model)
  - Window hands food back (FastAPI returns the score)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predict import predict_fraud   # import our prediction function
import traceback

# ─────────────────────────────────────────────
# 1. CREATE THE APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Taamine NLP Fraud Detection",
    description="Analyzes maintenance reports and claim descriptions for fraud indicators",
    version="1.0.0"
)

# Allow NestJS (running on a different port) to talk to this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # In production, change this to your NestJS URL
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# 2. DEFINE WHAT THE REQUEST LOOKS LIKE
#    (What data NestJS will send us)
# ─────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    claim_description: str          # What the company wrote about the incident
    maintenance_report_text: str    # Text extracted from their maintenance PDF
    claim_id: str | None = None     # Optional: the claim ID for logging


# ─────────────────────────────────────────────
# 3. DEFINE WHAT THE RESPONSE LOOKS LIKE
#    (What we send back to NestJS)
# ─────────────────────────────────────────────
class AnalyzeResponse(BaseModel):
    nlp_fraud_score:    float       # 0-100, the main score
    label:              str         # "FRAUD" or "LEGITIMATE"
    confidence:         float       # how sure the model is
    claim_score:        float       # score for the claim description alone
    maintenance_score:  float       # score for the maintenance report alone
    combined_score:     float       # final combined score
    flagged_signals:    list[str]   # list of suspicious phrases found


# ─────────────────────────────────────────────
# 4. HEALTH CHECK ENDPOINT
#    Your teammate can call GET /health to check
#    if the service is alive
# ─────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "nlp-fraud-detection",
        "message": "NLP model is loaded and ready"
    }


# ─────────────────────────────────────────────
# 5. THE MAIN ENDPOINT
#    POST /analyze-text
#    This is where the magic happens
# ─────────────────────────────────────────────
@app.post("/analyze-text", response_model=AnalyzeResponse)
def analyze_text(request: AnalyzeRequest):
    """
    Receives the claim description and maintenance report text.
    Returns a fraud score from 0 to 100.
    
    HOW TO CALL THIS (example):
    
    POST http://localhost:8001/analyze-text
    Content-Type: application/json
    
    {
        "claim_id": "7823",
        "claim_description": "Sudden failure with no prior warning...",
        "maintenance_report_text": "Monthly maintenance performed on..."
    }
    """
    
    try:
        # ── Step 1: Analyze the claim description alone ──
        claim_result = predict_fraud(request.claim_description)
        
        # ── Step 2: Analyze the maintenance report alone ──
        maintenance_result = predict_fraud(request.maintenance_report_text)
        
        # ── Step 3: Analyze both texts combined ──
        combined_text = f"""
        CLAIM DESCRIPTION:
        {request.claim_description}
        
        MAINTENANCE REPORT:
        {request.maintenance_report_text}
        """
        combined_result = predict_fraud(combined_text)
        
        # ── Step 4: Calculate the final weighted score ──
        # Give more weight to the combined analysis
        final_score = round(
            (claim_result["fraud_score"] * 0.30) +
            (maintenance_result["fraud_score"] * 0.30) +
            (combined_result["fraud_score"] * 0.40),
            1
        )
        
        # ── Step 5: Find suspicious phrases ──
        flagged = find_suspicious_phrases(
            request.claim_description + " " + request.maintenance_report_text
        )
        
        # ── Step 6: Determine final label ──
        label = "FRAUD" if final_score >= 50 else "LEGITIMATE"
        
        return AnalyzeResponse(
            nlp_fraud_score=    final_score,
            label=              label,
            confidence=         combined_result["confidence"],
            claim_score=        claim_result["fraud_score"],
            maintenance_score=  maintenance_result["fraud_score"],
            combined_score=     combined_result["fraud_score"],
            flagged_signals=    flagged,
        )

    except Exception as e:
        # If something goes wrong, return a clear error message
        print(f"❌ Error analyzing claim {request.claim_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ─────────────────────────────────────────────
# 6. SUSPICIOUS PHRASE DETECTOR
#    A simple keyword scanner that flags
#    common phrases found in fraudulent claims
# ─────────────────────────────────────────────
SUSPICIOUS_PHRASES = [
    # Phrases common in fake claims
    "sudden failure with no warning",
    "no prior indication",
    "perfectly maintained",
    "unexpected breakdown",
    "no signs of wear",
    "abrupt stop",
    "without any prior symptoms",
    "routine maintenance always performed",
    "all checks passed",
    "in perfect condition",
    "no anomalies detected before",
    # French versions (Algerian companies often write in French)
    "panne soudaine sans avertissement",
    "aucun signe préalable",
    "parfaitement entretenu",
    "arrêt brutal",
    "aucune anomalie",
]

def find_suspicious_phrases(text: str) -> list[str]:
    """
    Scans the text for known fraud indicator phrases.
    Returns a list of phrases that were found.
    """
    text_lower = text.lower()
    found = []
    for phrase in SUSPICIOUS_PHRASES:
        if phrase.lower() in text_lower:
            found.append(f"Suspicious phrase detected: '{phrase}'")
    return found


# ─────────────────────────────────────────────
# 7. RUN THE SERVER
#    When you run: python main.py
#    The server starts on port 8001
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
