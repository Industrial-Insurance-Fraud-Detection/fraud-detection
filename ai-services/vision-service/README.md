# vision-service

FastAPI microservice — **YOLOv8 damage detection + ELA manipulation analysis + EXIF forensics**.

Part of the Taamine AI pipeline. Port **8004**.

---

## Structure

```
vision-service/
├── app/
│   ├── main.py          # FastAPI app, lifespan, routes
│   ├── config.py        # All settings via env vars
│   ├── schemas.py       # Request / Response Pydantic models
│   ├── models/
│   │   └── yolov8.pt    # ← copy your trained weights here
│   └── services/
│       └── analyzer.py  # YOLOv8 + ELA + EXIF logic
├── tests/
│   ├── conftest.py      # Shared fixtures, image generators, mock YOLO
│   ├── test_api.py      # HTTP endpoint contract tests
│   ├── test_ela.py      # ELA unit tests
│   ├── test_exif.py     # EXIF unit tests
│   ├── test_detection.py# Detection scoring unit tests
│   └── test_analyzer.py # Full analyzer integration tests
├── Dockerfile
├── Dockerfile.gpu
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Quickstart

### 1. Copy model weights

```bash
# Copy best.pt from Kaggle output
cp /path/to/best.pt app/models/yolov8.pt
```

### 2. Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8004
```

### 3. Run with Docker

```bash
docker build -t vision-service .
docker run -p 8004:8004 \
  -v $(pwd)/app/models:/app/app/models \
  vision-service
```

---

## API

### `GET /health`
```json
{ "status": "ok", "model_loaded": true, "service": "vision-service" }
```

### `POST /analyze`

**Request:**
```json
{
  "photoPaths": ["https://minio.../photo1.jpg", "https://minio.../photo2.jpg"],
  "declaredDamage": "scratches",
  "incidentDate": "2026-02-10T00:00:00",
  "claimId": "SIN-2026-007823"
}
```

**Response:**
```json
{
  "score": 72.5,
  "boxes": [
    { "imageIndex": 0, "className": "patches", "confidence": 0.88,
      "x1": 10, "y1": 20, "x2": 100, "y2": 110 }
  ],
  "manipulation": false,
  "exifIssues": [
    { "imageIndex": 0, "issueType": "DATE_MISMATCH",
      "detail": "Photo taken 250 days before incident date",
      "severity": "CRITICAL" }
  ],
  "breakdown": { "detection": 80.0, "ela": 5.0, "exif": 100.0 },
  "indicators": ["DAMAGE_TYPE_MISMATCH", "DATE_MISMATCH"],
  "processingMs": 312
}
```

---

## Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=term-missing
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `app/models/yolov8.pt` | Path to YOLOv8 weights |
| `CONF_THRESHOLD` | `0.35` | Detection confidence cutoff |
| `ELA_FRAUD_THRESHOLD` | `35.0` | ELA score above this = manipulation |
| `EXIF_DATE_THRESHOLD_DAYS` | `30` | Days difference before EXIF flagged |
| `WEIGHT_DETECTION` | `0.45` | Detection component weight |
| `WEIGHT_ELA` | `0.30` | ELA component weight |
| `WEIGHT_EXIF` | `0.25` | EXIF component weight |
| `DEBUG` | `false` | Enable debug logging |