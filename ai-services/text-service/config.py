import os

SERVICE_NAME    = "text-service"
SERVICE_VERSION = "1.0.0"
PORT            = int(os.getenv("PORT", "8003"))

MODEL_PATH      = os.getenv("MODEL_PATH", "./model")
MAX_LENGTH      = int(os.getenv("MAX_LENGTH", "512"))

# Fraud score threshold — above this = FRAUD label
FRAUD_THRESHOLD = float(os.getenv("FRAUD_THRESHOLD", "50.0"))

# Suspicious phrases (French + English — Algerian industrial context)
SUSPICIOUS_PHRASES = [
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
    "panne soudaine sans avertissement",
    "aucun signe préalable",
    "parfaitement entretenu",
    "arrêt brutal",
    "aucune anomalie",
]