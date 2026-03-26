"""
consumer.py — RabbitMQ worker for anomaly-service.

Listens on the queue defined in config (default: anomaly-analysis-queue).
When a message arrives:
  1. Downloads the sensor CSV from MinIO
  2. Runs Isolation Forest + LSTM anomaly scoring
  3. Posts results back to the orchestrator via HTTP callback

This file is only used when running in async queue mode.
The orchestrator can also call POST /detect-anomalies directly (sync mode).

Start with:
    python -m app.consumer
"""

import asyncio
import io
import json
import logging
import signal
import sys

import aio_pika
import httpx
import pandas as pd
from minio import Minio

from app.config import (
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY,
    MINIO_BUCKET, MODEL_DIR,
)
from app.scorer import compute_anomaly_score

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)

# ── RabbitMQ / callback settings (mirror vision-service pattern) ──────────────
import os
RABBITMQ_URL    = os.getenv("RABBITMQ_URL",    "amqp://guest:guest@rabbitmq:5672/")
ANOMALY_QUEUE   = os.getenv("ANOMALY_QUEUE",   "anomaly-analysis-queue")
CALLBACK_URL    = os.getenv("CALLBACK_URL",    "http://backend:4000/internal/ai-results")


def _get_minio() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


def _load_models() -> dict:
    """Load all model artifacts — called once at worker startup."""
    import joblib
    import tensorflow as tf

    models = {
        "iso_forest":  joblib.load(f"{MODEL_DIR}/isolation_forest.pkl"),
        "scaler":      joblib.load(f"{MODEL_DIR}/scaler.pkl"),
        "sensor_cols": joblib.load(f"{MODEL_DIR}/sensor_cols.pkl"),
        "threshold":   joblib.load(f"{MODEL_DIR}/lstm_threshold.pkl"),
        "if_bounds":   joblib.load(f"{MODEL_DIR}/if_bounds.pkl"),
        "autoencoder": tf.keras.models.load_model(f"{MODEL_DIR}/lstm_autoencoder.keras"),
    }
    logger.info("Anomaly models loaded successfully")
    return models


async def handle_message(message: aio_pika.IncomingMessage, models: dict):
    """
    Process one anomaly analysis job from the queue.

    Expected message body (JSON):
    {
        "claimId":    "SIN-2026-007823",
        "csvPath":    "claims/sensor_data.csv",
        "claimDate":  "2026-02-10",
        "callbackUrl": "http://backend:4000/internal/ai-results"  (optional)
    }
    """
    async with message.process(requeue=True):
        body = {}
        try:
            body        = json.loads(message.body.decode())
            claim_id    = body.get("claimId")
            csv_path    = body.get("csvPath")
            claim_date  = body.get("claimDate")
            callback_url = body.get("callbackUrl", CALLBACK_URL)

            logger.info("Received job claimId=%s csvPath=%s", claim_id, csv_path)

            # ── download CSV from MinIO ───────────────────────────────────────
            minio_client = _get_minio()
            response     = minio_client.get_object(MINIO_BUCKET, csv_path)
            csv_bytes    = response.read()

            # ── parse CSV ─────────────────────────────────────────────────────
            df          = pd.read_csv(io.BytesIO(csv_bytes))
            sensor_cols = models["sensor_cols"]
            available   = [c for c in sensor_cols if c in df.columns]

            if len(available) < 5:
                raise ValueError(f"CSV has only {len(available)} sensor columns — minimum 5 required")

            if "timestamp" not in df.columns:
                raise ValueError("CSV missing required 'timestamp' column")

            timestamps    = df["timestamp"].reset_index(drop=True)
            df[available] = df[available].ffill().bfill()

            scaler        = models["scaler"]
            df[available] = scaler.transform(df[available])
            sensor_data   = df[available].reset_index(drop=True)

            # ── run scoring ───────────────────────────────────────────────────
            result = compute_anomaly_score(
                sensor_data, timestamps, claim_date, models
            )

            # ── post result back to orchestrator ──────────────────────────────
            payload = {
                "claimId":             claim_id,
                "anomalyScore":        result["score"],
                "anomalies":           result["anomalies"],
                "preIncidentAnomaly":  result["pre_incident_anomaly"],
                "fraudIndicator":      result["fraud_indicator"],
            }

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(callback_url, json=payload)
                resp.raise_for_status()
                logger.info("Result posted for claimId=%s  HTTP %d", claim_id, resp.status_code)

        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in message: %s", exc)
            await message.reject(requeue=False)

        except Exception as exc:
            logger.exception("Error processing claimId=%s: %s", body.get("claimId"), exc)
            raise


async def main():
    logger.info("Loading anomaly models...")
    models = _load_models()

    logger.info("Connecting to RabbitMQ at %s", RABBITMQ_URL)
    connection = await aio_pika.connect_robust(
        RABBITMQ_URL,
        reconnect_interval=5,
    )

    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=1)

        queue = await channel.declare_queue(
            ANOMALY_QUEUE,
            durable=True,
        )

        logger.info("Listening on queue: %s", ANOMALY_QUEUE)
        logger.info("Press Ctrl+C to stop.")

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(connection)))

        # pass models into the handler via lambda
        await queue.consume(lambda msg: handle_message(msg, models))
        await asyncio.Future()


async def _shutdown(connection):
    logger.info("Shutting down anomaly consumer...")
    await connection.close()
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())