# mock_ai.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sys

app = FastAPI()
SCORE = int(sys.argv[1]) if len(sys.argv) > 1 else 50

@app.post("/detect-anomalies")
def anomaly(body: dict = None):
    return {"score": SCORE, "fraud_indicator": "none", "anomalies": []}

@app.post("/classify-json")
def classify(body: dict = None):
    return {"fraud_score": SCORE, "predicted_class": "normal", "class_distribution": {}}

@app.post("/analyze-text")
def nlp(body: dict = None):
    return {"score": SCORE, "label": "legitimate", "flaggedSignals": [], "claimScore": SCORE, "maintenanceScore": SCORE}

@app.post("/analyze")
def vision(body: dict = None):
    return {"score": SCORE, "manipulation": False, "exifIssues": [], "boxes": []}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(sys.argv[2]) if len(sys.argv) > 2 else 8001)