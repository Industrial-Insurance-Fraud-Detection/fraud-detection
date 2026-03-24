from pydantic import BaseModel
from typing import List

class AnalyzeRequest(BaseModel):
    csvPath: str
    claimId: str
    claimDate: str  # "2026-02-10"

class AnomalyItem(BaseModel):
    timestamp: str
    parameter: str
    value: float
    threshold: float

class AnalyzeResponse(BaseModel):
    score: int
    anomalies: List[AnomalyItem]
    pre_incident_anomaly: bool
    fraud_indicator: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool