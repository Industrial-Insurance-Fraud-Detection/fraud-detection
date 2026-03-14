import os

# ── Service ──────────────────────────────────────────────────
SERVICE_NAME    = "classification-service"
SERVICE_VERSION = "1.0.0"
SERVICE_PORT    = int(os.getenv("SERVICE_PORT", 8002))

# ── Model paths ──────────────────────────────────────────────
MODEL_PATH              = os.getenv("MODEL_PATH", "models/classifier.pkl")
FEATURE_IMPORTANCE_PATH = os.getenv("FEATURE_IMPORTANCE_PATH", "models/feature_importance.png")

# ── Fraud score mapping ──────────────────────────────────────
CLASS_FRAUD_SCORES = {
    "FAKE":         90,
    "SABOTAGE":     85,
    "REAL_FAILURE": 20,
    "NORMAL_WEAR":  10,
}

# ── Acceptance criteria ──────────────────────────────────────
MIN_ACCURACY  = 0.80
MIN_PRECISION = 0.75
MIN_RECALL    = 0.80
