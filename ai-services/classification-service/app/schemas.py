from pydantic import BaseModel, Field
from typing import Dict, Optional


# ── Input Schemas ────────────────────────────────────────────

class MaintenanceInput(BaseModel):
    Type: str = Field(..., description="Machine type: L, M, or H")
    Air_temperature: float = Field(..., description="Air temperature in Kelvin")
    Process_temperature: float = Field(..., description="Process temperature in Kelvin")
    Rotational_speed: float = Field(..., description="Rotational speed in RPM")
    Torque: float = Field(..., description="Torque in Nm")
    Tool_wear: float = Field(..., description="Tool wear in minutes")

    class Config:
        json_schema_extra = {
            "example": {
                "Type": "M",
                "Air_temperature": 298.1,
                "Process_temperature": 308.6,
                "Rotational_speed": 1551.0,
                "Torque": 42.8,
                "Tool_wear": 0.0
            }
        }


# ── Output Schemas ───────────────────────────────────────────

class ClassificationResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class: FAKE, NORMAL_WEAR, REAL_FAILURE, SABOTAGE")
    fraud_score: int = Field(..., description="Fraud score from 0 to 100")
    confidence: float = Field(..., description="Model confidence (0-1)")
    input_received: dict
    service: str


class ClassifyFailureResponse(BaseModel):
    predicted_class: str = Field(..., description="Majority predicted class")
    fraud_score: int = Field(..., description="Fraud score from 0 to 100")
    row_count: int = Field(..., description="Number of rows analyzed")
    class_distribution: Dict[str, int]
    feature_importance: Dict[str, float]
    feature_importance_chart_b64: Optional[str] = None
    model: str = "XGBoost"
    service: str


class TrainResponse(BaseModel):
    status: str
    accuracy: float
    precision: float
    recall: float


class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
