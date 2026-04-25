# Taamine — Project Setup Guide

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Node.js | ≥ 18 | NestJS backend |
| Python | ≥ 3.10 | FastAPI AI services |
| Docker Desktop | latest | RabbitMQ, Redis, PostgreSQL, MinIO |
| n8n | latest | Workflow automation |

---

## 1. Infrastructure (Docker)

Start all infrastructure services with a single command. Create `docker-compose.infra.yml` in the project root:

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: taamine
      POSTGRES_PASSWORD: taamine_pass
      POSTGRES_DB: taamine_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"    # AMQP
      - "15672:15672"  # Management UI (guest/guest)
    environment:
      RABBITMQ_DEFAULT_USER: guest
      RABBITMQ_DEFAULT_PASS: guest

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports:
      - "9000:9000"    # API
      - "9001:9001"    # Console UI
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data

volumes:
  postgres_data:
  minio_data:
```

```bash
docker compose -f docker-compose.infra.yml up -d
```

**Verify:**
- RabbitMQ UI → http://localhost:15672 (guest/guest)
- MinIO Console → http://localhost:9001 (minioadmin/minioadmin)

---

## 2. NestJS Backend

### Environment
Create `backend/.env`:

```env
# Database
DATABASE_URL="postgresql://taamine:taamine_pass@localhost:5432/taamine_db"

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# JWT
JWT_ACCESS_SECRET=your-access-secret-change-this
JWT_REFRESH_SECRET=your-refresh-secret-change-this
JWT_ACCESS_EXPIRATION=15m
JWT_REFRESH_EXPIRATION=7d

# RabbitMQ
RABBITMQ_URL=amqp://guest:guest@localhost:5672

# MinIO
MINIO_ENDPOINT=localhost
MINIO_PORT=9000
MINIO_USE_SSL=false
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=claims

# AI Services
AI_ANOMALY_URL=http://localhost:8001
AI_CLASSIFICATION_URL=http://localhost:8002
AI_NLP_URL=http://localhost:8003
AI_VISION_URL=http://localhost:8004

# n8n
N8N_WEBHOOK_BASE=http://localhost:5678/webhook

# Security
BCRYPT_ROUNDS=12
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=30

# App
PORT=3000
FRONTEND_URL=http://localhost:5173
```

### Run

```bash
cd backend
npm install
npx prisma migrate dev --name init
npx prisma generate
npm run start:dev
```

API: http://localhost:3000/api/v1  
Swagger: http://localhost:3000/api/docs

---

## 3. AI Microservices

Each service needs model files present before starting. Your teammates export `.pkl`, `.keras`, `.pt` files from Kaggle and place them in the correct `models/` folder.

### anomaly-service (port 8001)

```bash
cd ai-services/anomaly-service
pip install -r requirements.txt

# Required model files in app/models/:
# isolation_forest.pkl, scaler.pkl, sensor_cols.pkl,
# lstm_threshold.pkl, if_bounds.pkl, lstm_autoencoder.keras

uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### classification-service (port 8002)

```bash
cd ai-services/classification-service
pip install -r requirements.txt

# Required: models/classifier.pkl

cd app
uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

### text-service (port 8003)

```bash
cd ai-services/text-service
pip install -r requirements.txt

# Required: ./model/ folder (HuggingFace BERT)
# Contains: config.json, pytorch_model.bin, tokenizer files

uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

### vision-service (port 8004)

```bash
cd ai-services/vision-service
pip install -r requirements.txt

# Required: app/models/yolov8.pt

uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
```

**Health check all services:**
```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
```

---

## 4. n8n Setup

### Option A — Local (Development)

```bash
# Install globally
npm install -g n8n

# Start
n8n start
```

n8n UI: http://localhost:5678

### Option B — Docker (Recommended for sharing)

```bash
docker run -d \
  --name n8n \
  -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  n8nio/n8n
```

### ⚠️ Sharing n8n Workflows with Others

n8n workflows are stored locally. To share them so others can run the project:

**Step 1 — Export your workflows:**
1. Open n8n UI → http://localhost:5678
2. Go to each workflow → top-right menu (⋯) → **Download**
3. This saves a `.json` file per workflow

**Step 2 — Add to the repo:**
```
project-root/
  n8n-workflows/
    approved-pdf-workflow.json
    rejected-pdf-workflow.json
```

**Step 3 — Import instructions for others:**
1. Start n8n
2. Open http://localhost:5678
3. Click **+** (New Workflow) → **Import from file**
4. Select the `.json` file
5. Click **Activate** (toggle top-right)
6. **Repeat for each workflow**

### ⚠️ Long-lived JWT Token for n8n HTTP Requests

n8n's HTTP Request nodes call backend endpoints. The default 15-minute access token expires during workflow execution. Use a long-lived token:

1. Register a dedicated n8n service account in your system:
```bash
curl -X POST http://localhost:3000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"firstName":"n8n","lastName":"Bot","email":"n8n@taamine.internal","password":"N8nBot2026!"}'
```

2. Note: for internal service calls (PDF generation, notifications), the backend endpoints use `@Public()` and **do not require a token at all**. No token needed for:
   - `POST /files/generate-and-save-pdf`
   - `POST /notifications/internal`
   - `PATCH /claims/:id/pdf-url`

---

## 5. Full Startup Order

Always start services in this order to avoid connection errors:

```
1. docker compose -f docker-compose.infra.yml up -d
   (PostgreSQL, Redis, RabbitMQ, MinIO)

2. cd backend && npm run start:dev
   (waits for DB → runs migrations on first start)

3. cd ai-services/anomaly-service && uvicorn app.main:app --port 8001
4. cd ai-services/classification-service/app && uvicorn main:app --port 8002
5. cd ai-services/text-service && uvicorn main:app --port 8003
6. cd ai-services/vision-service && uvicorn app.main:app --port 8004

7. n8n start
   → Import workflows from n8n-workflows/ if first time
   → Activate all workflows
```

---

## 6. Verify Everything Works

```bash
# 1. Health check
curl http://localhost:3000/api/v1/health

# 2. Register a test client (PowerShell: use curl.exe)
curl.exe -X POST http://localhost:3000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d "{\"firstName\":\"Test\",\"lastName\":\"User\",\"email\":\"test@test.dz\",\"password\":\"Test1234\"}"

# 3. Login
curl.exe -X POST http://localhost:3000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d "{\"email\":\"test@test.dz\",\"password\":\"Test1234\"}"

# 4. Run tests
cd backend && npm test
```

---

## 7. Model Files Checklist

Before the project can run full end-to-end analysis, these files must exist:

| Service | File(s) | Source |
|---------|---------|--------|
| anomaly-service | `app/models/isolation_forest.pkl` | Teammate (Kaggle) |
| anomaly-service | `app/models/scaler.pkl` | Teammate (Kaggle) |
| anomaly-service | `app/models/sensor_cols.pkl` | Teammate (Kaggle) |
| anomaly-service | `app/models/lstm_threshold.pkl` | Teammate (Kaggle) |
| anomaly-service | `app/models/if_bounds.pkl` | Teammate (Kaggle) |
| anomaly-service | `app/models/lstm_autoencoder.keras` | Teammate (Kaggle) |
| classification-service | `models/classifier.pkl` | Teammate (Kaggle) |
| text-service | `model/` (full HuggingFace folder) | Teammate (Kaggle) |
| vision-service | `app/models/yolov8.pt` | Teammate (Kaggle) |

Without model files, services start in **degraded mode** (health returns `model_loaded: false`) and the worker falls back to score 50 for those services — the pipeline still runs, just without real AI scoring.

---

## 8. Ports Summary

| Service | Port | URL |
|---------|------|-----|
| NestJS API | 3000 | http://localhost:3000/api/v1 |
| Swagger | 3000 | http://localhost:3000/api/docs |
| anomaly-service | 8001 | http://localhost:8001/docs |
| classification-service | 8002 | http://localhost:8002/docs |
| text-service | 8003 | http://localhost:8003/docs |
| vision-service | 8004 | http://localhost:8004/docs |
| n8n | 5678 | http://localhost:5678 |
| PostgreSQL | 5432 | — |
| Redis | 6379 | — |
| RabbitMQ AMQP | 5672 | — |
| RabbitMQ UI | 15672 | http://localhost:15672 |
| MinIO API | 9000 | — |
| MinIO Console | 9001 | http://localhost:9001 |
