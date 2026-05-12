# Taamine — Backend

NestJS REST API + AI orchestration worker for the Taamine fraud detection platform.

---

## Stack

- **NestJS** (TypeScript) — modular, decorator-driven framework
- **Prisma ORM** — type-safe database client
- **PostgreSQL 16** — primary data store
- **Redis 7** — JWT blacklist, refresh token sessions, account lockout
- **RabbitMQ 3.12** — decouples claim submission from AI analysis
- **MinIO** — object storage for sensor CSVs, photos, and decision PDFs

---

## Module Overview

| Module | Responsibility |
|---|---|
| `auth` | Registration, login, JWT issuance, refresh token rotation, logout, password reset |
| `users` | Profile management (client + investigator) |
| `equipment` | Industrial machine registry — CRUD with soft delete |
| `claims` | Claim submission, state machine, auto/human decision |
| `files` | MinIO upload, presigned URL generation, PDF generation |
| `queue` | RabbitMQ producer + AI orchestration worker |
| `notifications` | In-app notification creation and read management |
| `common` | Redis service, audit service, guards, interceptors, filters |
| `health` | `GET /health` — postgres + redis + memory checks |
| `prisma` | Global PrismaClient singleton |

---

## Security Architecture

### JWT Strategy

- **Access tokens** — 15-minute expiry, signed with `JWT_ACCESS_SECRET`
- **Refresh tokens** — 7-day expiry, signed with `JWT_REFRESH_SECRET`, stored in Redis as `session:{userId}:{jti}`
- **Blacklisting** — on logout, the access token's `jti` is written to Redis with TTL = remaining lifetime. The `JwtStrategy` checks the blacklist on every request.
- **Token rotation** — every refresh call deletes the old session and issues a new token pair. Reuse of an already-used refresh token triggers immediate revocation of all sessions for that user (reuse attack detection).

### Account Lockout

- Failed login attempts are tracked in Redis as `attempts:{userId}`
- After 5 failures (configurable), a `lockout:{userId}` key is set with a 30-minute TTL
- The remaining lockout time is returned in the error message

### Role-Based Access Control

Two roles: `CLIENT` and `INVESTIGATOR`.

- `@Roles(Role.CLIENT)` — clients only
- `@Roles(Role.INVESTIGATOR)` — investigators only
- No decorator — any authenticated user
- `@Public()` — bypass JWT entirely (used for n8n internal callbacks)

**Ownership pattern:** clients requesting another user's claim receive `404 Not Found`, not `403 Forbidden`. This prevents claim ID enumeration.

---

## AI Orchestration Worker

Located in `src/queue/queue.worker.ts`. Consumes `ai-analysis` messages from RabbitMQ.

```
Claim submitted
      │
      ▼
Status → ANALYZING
      │
      ▼ Promise.allSettled (all 4 in parallel, 60s timeout each)
┌─────┴──────────────────────────────────────────────────┐
│ Anomaly (8001)  Classification (8002)  NLP (8003)  Vision (8004) │
└─────┬──────────────────────────────────────────────────┘
      │ Failed service → neutral score 50
      ▼
Weighted score = anomaly×0.35 + class×0.25 + nlp×0.20 + vision×0.20
      │
      ├─ score < 30  → AUTO_APPROVED  → n8n webhook → PDF → notify client
      ├─ score 30–69 → HUMAN_REVIEW   → notify client + notify investigators
      └─ score ≥ 70  → AUTO_REJECTED  → n8n webhook → PDF → notify client
```

**Graceful degradation:** `Promise.allSettled` ensures that if any AI service is down, the claim is not lost. The failed service contributes a neutral score of 50. If all four fail, the final score is 50 → routed to human review.

---

## Running Locally

### Prerequisites

- Node.js 20+
- PostgreSQL running on port 5432
- Redis running on port 6379
- RabbitMQ running on port 5672
- MinIO running on port 9000

### Setup

```bash
cd backend
cp .env.example .env
# Fill in .env values

npm install --legacy-peer-deps
npx prisma migrate dev
npm run start:dev
```

### Environment Variables

```env
# Database
DATABASE_URL=postgresql://taamine:secret@localhost:5432/taamine

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# RabbitMQ
RABBITMQ_URL=amqp://guest:guest@localhost:5672

# JWT
JWT_ACCESS_SECRET=your-access-secret
JWT_REFRESH_SECRET=your-refresh-secret
JWT_ACCESS_EXPIRATION=15m
JWT_REFRESH_EXPIRATION=7d

# MinIO
MINIO_ENDPOINT=localhost
MINIO_PORT=9000
MINIO_USE_SSL=false
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET=taamine-files

# AI Services
AI_ANOMALY_URL=http://localhost:8001
AI_CLASSIFICATION_URL=http://localhost:8002
AI_NLP_URL=http://localhost:8003
AI_VISION_URL=http://localhost:8004

# n8n
N8N_WEBHOOK_BASE=http://localhost:5678/webhook

# App
PORT=3000
FRONTEND_URL=http://localhost:5173
BCRYPT_ROUNDS=12
MAX_LOGIN_ATTEMPTS=5
LOCKOUT_DURATION_MINUTES=30
```

---

## Running with Docker

```bash
# From project root
docker compose up backend
```

The backend service waits for postgres, rabbitmq, redis, and minio to be healthy before starting.

---

## API Reference

Full interactive docs available at `http://localhost:3000/api/docs` (Swagger UI).

### Auth

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | `/auth/register` | — | Create client account |
| POST | `/auth/login` | — | Login, receive token pair |
| POST | `/auth/refresh` | — | Rotate refresh token |
| POST | `/auth/logout` | JWT | Blacklist current session |
| POST | `/auth/logout-all` | JWT | Revoke all sessions |
| POST | `/auth/change-password` | JWT | Change password |
| POST | `/auth/forgot-password` | — | Generate reset token |
| POST | `/auth/reset-password` | — | Reset with token |

### Claims

| Method | Endpoint | Role | Description |
|---|---|---|---|
| POST | `/claims` | CLIENT | Submit claim (multipart: CSV + photos + optional PDF) |
| GET | `/claims/my` | CLIENT | My claims (paginated) |
| GET | `/claims/flagged` | INVESTIGATOR | HUMAN_REVIEW queue sorted by score |
| GET | `/claims/:id` | Any | Full claim detail |
| PATCH | `/claims/:id/decide` | INVESTIGATOR | Submit APPROVED/REJECTED decision |
| PATCH | `/claims/:id/pdf-url` | Public | Called by n8n to save decision PDF URL |

### Equipment

| Method | Endpoint | Role | Description |
|---|---|---|---|
| POST | `/equipment` | CLIENT | Register machine |
| GET | `/equipment` | CLIENT | My machines (paginated, search, type filter) |
| GET | `/equipment/:id` | CLIENT | Machine detail + last 5 claims |
| PATCH | `/equipment/:id` | CLIENT | Update info |
| DELETE | `/equipment/:id` | CLIENT | Soft delete (isActive = false) |

---

## Tests

```bash
npm run test              # Run all unit tests
npm run test -- --watch   # Watch mode
npm run test -- --coverage
```

**87 passing tests** across 7 modules. All tests use Jest with fully mocked dependencies — no real database, Redis, or RabbitMQ connections needed.

Test files:
- `auth.service.spec.ts` — 22 tests
- `claims.service.spec.ts` — 17 tests
- `equipment.service.spec.ts` — 16 tests
- `files.service.spec.ts` — 5 tests
- `notifications.service.spec.ts` — 9 tests
- `users.service.spec.ts` — 8 tests
- `queue.worker.spec.ts` — 10 tests

---

## Claim State Machine

```
PENDING → ANALYZING → HUMAN_REVIEW → APPROVED
                    →              → REJECTED
                    → APPROVED (auto)
                    → REJECTED (auto)
```

State transitions are enforced in `ClaimsService.submitDecision()` — a claim can only be decided when in `HUMAN_REVIEW` status.
