"""
Configuration for vision-service.
All values are read from environment variables with safe defaults.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── service ───────────────────────────────────────────────────────────────
    SERVICE_NAME: str = "vision-service"
    PORT: int = 8004
    DEBUG: bool = False

    # ── model ─────────────────────────────────────────────────────────────────
    MODEL_PATH: str = "app/models/yolov8.pt"
    CONF_THRESHOLD: float = 0.35
    IOU_THRESHOLD: float = 0.45

    CLASS_NAMES: list[str] = [
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled-in_scale", "scratches",
    ]

    # ── fraud thresholds ──────────────────────────────────────────────────────
    DETECTION_FRAUD_THRESHOLD: float = 70.0
    ELA_FRAUD_THRESHOLD: float = 35.0
    EXIF_DATE_THRESHOLD_DAYS: int = 30

    # ── vision score weights (must sum to 1.0) ────────────────────────────────
    WEIGHT_DETECTION: float = 0.45
    WEIGHT_ELA: float = 0.30
    WEIGHT_EXIF: float = 0.25

    # ── ELA signature (must match Kaggle training notebook) ───────────────────
    ELA_SIGNATURE_VALUE: int = 42
    ELA_SIGNATURE_ROWS: list[int] = [0, 2, 4, 6, 8, 10, 12, 14]

    # ── RabbitMQ (used by consumer.py) ────────────────────────────────────────
    RABBITMQ_URL: str = "amqp://guest:guest@rabbitmq:5672/"
    VISION_QUEUE: str = "vision-analysis-queue"
    CALLBACK_URL: str = "http://backend:4000/internal/ai-results"

    model_config = {"env_file": ".env", "protected_namespaces": ()}


settings = Settings()