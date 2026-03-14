# ⚙️ Backend — Industrial Insurance Fraud Detection

> **Status: Work in Progress 🛠️**

This directory houses the core business logic and orchestration engine for the fraud detection system.

## 🏗️ Planned Tech Stack
- **Framework**: [NestJS](https://nestjs.com/)
- **Language**: TypeScript
- **ORM**: [Prisma](https://www.prisma.io/)
- **Database**: MySQL / PostgreSQL (Docker-based)
- **Task Queue**: [BullMQ](https://docs.bullmq.io/) (Redis-backed)

## 🚀 Key Responsibilities
- **AI Job Scheduling**: Managing requests to the 4 specialized AI microservices.
- **Claim Lifecycle**: Handling CRUD operations for insurance claims.
- **Data Aggregation**: Combining AI scores into a final fraud probability.
- **User Authentication**: Secure access for authorized investigators.

## ⚙️ Development Setup
*(Instructions will be added when the initial boilerplate is committed)*
```bash
npm install
npm run start:dev
```
