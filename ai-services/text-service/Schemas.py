from pydantic import BaseModel
from typing import List, Optional


# ── Request ───────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    claimId:                  Optional[str] = None
    claimDescription:         str
    maintenanceReportText:    str

    class Config:
        json_schema_extra = {
            "example": {
                "claimId": "SIN-2026-001",
                "claimDescription": "Sudden failure with no prior warning.",
                "maintenanceReportText": "Monthly maintenance performed on schedule."
            }
        }


# ── Response ──────────────────────────────────────────────────

class AnalyzeResponse(BaseModel):
    score:            float        # 0-100 final weighted NLP fraud score
    label:            str          # FRAUD | LEGITIMATE
    confidence:       float        # model confidence 0-100
    claimScore:       float        # score for claim description alone
    maintenanceScore: float        # score for maintenance report alone
    combinedScore:    float        # score for both texts combined
    flaggedSignals:   List[str]    # suspicious phrases detected


# ── Health ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str
    service:      str
    modelLoaded:  bool