# Taamine — AI-Powered Industrial Insurance Fraud Detection

> Graduation project — Université M'Hamed Bougara de Boumerdès  
> Supervised by Dr. Yahiatene

---

## The Problem

Industrial insurance fraud is a critical and growing challenge in Algeria's heavy industry sector. Companies like Sonatrach, Sonelgaz, and their supply chain operate fleets of expensive industrial equipment — pumps, compressors, turbines, motors — worth hundreds of millions of dinars. When a machine fails, the insurance claim process is entirely manual: an investigator reads a report, looks at a few photos, and makes a judgment call.

This creates three compounding problems:

**1. Human judgment is inconsistent and slow.** A single investigator can only review so many dossiers per day. Backlogs grow. Legitimate claims wait weeks. Fraudulent ones slip through due to cognitive overload.

**2. Fraud is multimodal and hard to spot manually.** A sophisticated fraud involves manipulated sensor data, a fabricated maintenance report, and photos taken months before the claimed incident. No single human reviewer checks all three simultaneously, let alone cross-references them against each other.

**3. There is no audit trail.** Decisions are verbal or on paper. There is no systematic record of why a claim was approved or rejected — which creates legal exposure and no feedback loop to improve detection over time.

The result: insurance companies absorb fraudulent payouts they cannot defend in court, while legitimate industrial clients wait too long for decisions on real equipment failures.

---

## The Solution

**Taamine** is an end-to-end fraud detection platform that combines four AI models operating in parallel to score every claim on a 0–100 fraud probability scale — automatically, in under 5 minutes.

The system analyzes three types of evidence simultaneously:

| Evidence Type | What It Detects |
|---|---|
| **Sensor CSV data** | Anomaly patterns and pre-incident signatures |
| **Maintenance report PDF** | Narrative inconsistencies and contradictions |
| **Equipment photos** | Image forgery, EXIF metadata manipulation |

Four specialized models each produce a score, which are combined into a single weighted fraud score:

```
Final Score = (Anomaly × 0.35) + (Classification × 0.25) + (NLP × 0.20) + (Vision × 0.20)
```

**Routing logic:**
- Score **< 30** → Auto-approved. Client notified instantly.
- Score **30–69** → Routed to a human investigator for review.
- Score **≥ 70** → Auto-rejected. Fraud report generated.

In all three cases, a PDF decision letter is generated automatically and stored in object storage.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        React Frontend                        │
│            Client Portal  |  Investigator Portal            │
└─────────────────────────────────┬───────────────────────────┘
                                  │ REST API
┌─────────────────────────────────▼───────────────────────────┐
│                     NestJS Backend (port 3000)               │
│  Auth · Users · Equipment · Claims · Files · Notifications  │
│  JWT Blacklisting · Refresh Rotation · Audit Logs · RBAC    │
└──────┬──────────────┬──────────────────────────────────┬────┘
       │ PostgreSQL   │ RabbitMQ                         │ Redis
       │ + Prisma     │                                  │ Sessions
       │              │                                  │ Blacklist
       │    ┌─────────▼─────────────────────────────┐   │
       │    │          AI Worker (NestJS)             │   │
       │    │  Promise.allSettled → 4 AI services    │   │
       │    └──────┬──────┬──────────┬──────┬────────┘   │
       │           │      │          │      │             │
       │    ┌──────▼┐ ┌───▼───┐ ┌───▼──┐ ┌─▼──────┐     │
       │    │:8001  │ │:8002  │ │:8003 │ │:8004   │     │
       │    │Anomaly│ │Class. │ │ NLP  │ │Vision  │     │
       │    │LSTM+IF│ │XGBoost│ │BERT  │ │YOLOv8  │     │
       │    └───────┘ └───────┘ └──────┘ └────────┘     │
       │                                                  │
       │         n8n (port 5678)                         │
       │    PDF Generation · Email Alerts · Webhooks     │
       │                                                  │
       │         MinIO (port 9000)                       │
       │    Sensor CSVs · Photos · Decision PDFs         │
       └──────────────────────────────────────────────────┘
```

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite, Zustand, Recharts, Three.js |
| Backend | NestJS (TypeScript), Prisma ORM |
| Database | PostgreSQL 16 |
| Cache / Sessions | Redis 7 |
| Message Queue | RabbitMQ 3.12 |
| Object Storage | MinIO |
| Workflow Automation | n8n |
| AI Services | Python FastAPI (×4) |
| AI Models | Isolation Forest + LSTM, XGBoost, BERT multilingual, YOLOv8 |

---

## Project Structure

```
taamine/
├── backend/                  # NestJS API + AI worker
│   ├── src/
│   │   ├── auth/             # JWT, refresh rotation, blacklist, lockout
│   │   ├── users/            # Profile management
│   │   ├── equipment/        # Machine registry
│   │   ├── claims/           # Claim lifecycle state machine
│   │   ├── files/            # MinIO integration, presigned URLs, PDF gen
│   │   ├── queue/            # RabbitMQ producer + AI orchestration worker
│   │   ├── notifications/    # In-app notification system
│   │   ├── common/           # Guards, Redis, Audit logs
│   │   └── health/           # /health endpoint
│   └── prisma/               # Schema + migrations
│
├── frontend/                 # React SPA
│   └── src/
│       ├── pages/client/     # Client dashboard, claims, equipment
│       ├── pages/investigator/ # Review queue, decision interface
│       └── pages/auth/       # Login, register, password reset
│
├── ai-services/
│   ├── anomaly-service/      # Port 8001 — LSTM + Isolation Forest
│   ├── classification-service/ # Port 8002 — XGBoost
│   ├── text-service/         # Port 8003 — BERT NLP
│   └── vision-service/       # Port 8004 — YOLOv8 + ELA + EXIF
│
├── docker-compose.yml        # Full stack orchestration
└── start-taamine.ps1         # Local dev startup (Windows)
```

---

## Quick Start (Docker)

```bash
# Clone and start everything
git clone <repo>
cd taamine

# Copy and configure environment
cp backend/.env.example backend/.env
# Edit backend/.env with your secrets

# Start all 9 services
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:80 |
| API | http://localhost:3000/api/v1 |
| Swagger | http://localhost:3000/api/docs |
| n8n | http://localhost:5678 |
| MinIO Console | http://localhost:9001 |
| RabbitMQ Console | http://localhost:15672 (guest/guest) |

---

## Quick Start (Local Dev — Windows)

```powershell
# From project root
powershell -ExecutionPolicy Bypass -File start-taamine.ps1
```

This opens 9 terminal windows, one per service.

---

## CI/CD

GitHub Actions pipeline at `.github/workflows/ci.yml`:

- **Backend** — type check + 87 Jest unit tests (with live postgres/redis/rabbitmq service containers)
- **Anomaly, Classification, Text, Vision** — ruff lint + pytest
- Triggers on push/PR to `main` and `develop`

---

## Key Design Decisions

**Graceful AI degradation** — The worker uses `Promise.allSettled`, not `Promise.all`. If any AI service is unreachable, it falls back to a neutral score of 50 for that model. The claim is never lost; it routes to human review instead of crashing.

**n8n for post-decision orchestration** — PDF generation, email alerts, and report archiving are handled by n8n webhooks, not embedded in the backend. This means new notification channels (SMS, Slack) can be added without touching backend code.

**404 not 403 for ownership** — Clients requesting another user's claim receive a 404, not a 403. This prevents attackers from discovering whether a claim ID exists.

**Real ELA for image forensics** — Error Level Analysis is implemented as pixel differential after JPEG recompression at quality=75, not as a steganographic signature trick. This is defensible in a legal context.

---


