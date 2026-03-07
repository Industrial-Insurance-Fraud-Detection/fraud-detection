"""
consumer.py — RabbitMQ worker for vision-service.

Listens on the queue defined in config (default: vision-analysis-queue).
When a message arrives:
  1. Downloads photos from MinIO signed URLs
  2. Runs YOLOv8 + ELA + EXIF analysis
  3. Posts results back to the orchestrator via HTTP

This file is only used when running in async queue mode.
The orchestrator can also call POST /analyze directly (sync mode).

Start with:
    python -m app.consumer
"""

import asyncio
import json
import logging
import signal
import sys

import aio_pika
import httpx

from app.config import settings
from app.schemas import AnalyzeRequest
from app.services.analyzer import analyzer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


async def handle_message(message: aio_pika.IncomingMessage):
    """
    Process one claim analysis job from the queue.

    Expected message body (JSON):
    {
        "claimId":       "SIN-2026-007823",
        "photoPaths":    ["https://minio.../photo1.jpg", ...],
        "declaredDamage": "scratches",
        "incidentDate":  "2026-02-10T00:00:00",
        "callbackUrl":   "http://backend:4000/internal/ai-results"
    }
    """
    async with message.process(requeue=True):
        try:
            body = json.loads(message.body.decode())
            claim_id     = body.get("claimId")
            photo_paths  = body.get("photoPaths", [])
            declared     = body.get("declaredDamage")
            incident_raw = body.get("incidentDate")
            callback_url = body.get("callbackUrl", settings.CALLBACK_URL)

            logger.info("Received job claimId=%s photos=%d", claim_id, len(photo_paths))

            # parse incident date if present
            incident_date = None
            if incident_raw:
                from datetime import datetime
                try:
                    incident_date = datetime.fromisoformat(incident_raw)
                except ValueError:
                    logger.warning("Could not parse incidentDate: %s", incident_raw)

            # run analysis
            result = analyzer.analyze(
                photo_paths=photo_paths,
                declared_damage=declared,
                incident_date=incident_date,
                claim_id=claim_id,
            )

            # send result back to NestJS backend
            payload = {
                "claimId":       claim_id,
                "visionScore":   result.score,
                "boxes":         [b.model_dump() for b in result.boxes],
                "manipulation":  result.manipulation,
                "exifIssues":    [e.model_dump() for e in result.exifIssues],
                "breakdown":     result.breakdown.model_dump(),
                "indicators":    result.indicators,
                "processingMs":  result.processingMs,
            }

            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(callback_url, json=payload)
                resp.raise_for_status()
                logger.info("Result posted for claimId=%s  HTTP %d", claim_id, resp.status_code)

        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON in message: %s", exc)
            # don't requeue malformed messages
            await message.reject(requeue=False)

        except Exception as exc:
            logger.exception("Error processing claimId=%s: %s", body.get("claimId"), exc)
            # message.process(requeue=True) will requeue automatically on exception
            raise


async def main():
    logger.info("Connecting to RabbitMQ at %s", settings.RABBITMQ_URL)

    connection = await aio_pika.connect_robust(
        settings.RABBITMQ_URL,
        reconnect_interval=5,
    )

    async with connection:
        channel = await connection.channel()

        # limit to 1 unacked message at a time — prevents overload
        await channel.set_qos(prefetch_count=1)

        queue = await channel.declare_queue(
            settings.VISION_QUEUE,
            durable=True,       # survives RabbitMQ restart
        )

        logger.info("Listening on queue: %s", settings.VISION_QUEUE)
        logger.info("Press Ctrl+C to stop.")

        # graceful shutdown on SIGTERM (Docker stop)
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(connection)))

        await queue.consume(handle_message)
        await asyncio.Future()   # run forever


async def _shutdown(connection):
    logger.info("Shutting down consumer...")
    await connection.close()
    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())