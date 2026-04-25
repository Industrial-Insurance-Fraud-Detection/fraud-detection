# ============================================================
# Taamine — Start All Services
# Run this script from the project root:
#   Right-click → Run with PowerShell
#   OR: powershell -ExecutionPolicy Bypass -File start-taamine.ps1
# ============================================================

$projectRoot = $PSScriptRoot

# ── Helper: open a new terminal window ──────────────────────
function Start-Service {
    param(
        [string]$Title,
        [string]$WorkDir,
        [string]$Command
    )
    Start-Process powershell -ArgumentList `
        "-NoExit", `
        "-Command", `
        "`$host.UI.RawUI.WindowTitle = '$Title'; cd '$WorkDir'; $Command"
}

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Starting Taamine Services..." -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Redis ─────────────────────────────────────────────────
Write-Host "[1/8] Starting Redis..." -ForegroundColor Yellow
Start-Service `
    -Title "Redis" `
    -WorkDir "C:\" `
    -Command "redis-server"

Start-Sleep -Seconds 1

# ── 2. RabbitMQ ──────────────────────────────────────────────
Write-Host "[2/8] Starting RabbitMQ..." -ForegroundColor Yellow
Start-Service `
    -Title "RabbitMQ" `
    -WorkDir "C:\" `
    -Command "rabbitmq-server"

Start-Sleep -Seconds 2

# ── 3. MinIO ─────────────────────────────────────────────────
Write-Host "[3/8] Starting MinIO..." -ForegroundColor Yellow
Start-Service `
    -Title "MinIO" `
    -WorkDir "C:\" `
    -Command "minio.exe server C:/minio-data --console-address ':9001'"

Start-Sleep -Seconds 1

# ── 4. NestJS Backend ────────────────────────────────────────
Write-Host "[4/8] Starting NestJS Backend..." -ForegroundColor Yellow
Start-Service `
    -Title "NestJS Backend" `
    -WorkDir "$projectRoot\backend" `
    -Command "npm run start:dev"

Start-Sleep -Seconds 2

# ── 5. Anomaly Service ───────────────────────────────────────
Write-Host "[5/8] Starting Anomaly Service (port 8001)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "
cd $projectRoot\ai-services\anomaly-service;
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
"

Start-Sleep -Seconds 1


# ── 6. Classification Service ────────────────────────────────
Write-Host "[6/8] Starting Classification Service (port 8002)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "
cd $projectRoot\ai-services\classification-service;
.\.venv\Scripts\Activate.ps1;
uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
"

Start-Sleep -Seconds 1


# ── 7. Text/NLP Service ──────────────────────────────────────
Write-Host "[7/8] Starting Text/NLP Service (port 8003)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "
cd $projectRoot\ai-services\text-service;
uvicorn main:app --host 0.0.0.0 --port 8003 --reload
"

Start-Sleep -Seconds 1


# ── 8. Vision Service ────────────────────────────────────────
Write-Host "[8/8] Starting Vision Service (port 8004)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "
cd $projectRoot\ai-services\vision-service;
uvicorn app.main:app --host 0.0.0.0 --port 8004 --reload
"

Start-Sleep -Seconds 1

# ── 9. n8n ───────────────────────────────────────────────────
Write-Host "[9/9] Starting n8n..." -ForegroundColor Yellow
Start-Service `
    -Title "n8n Workflows" `
    -WorkDir "C:\" `
    -Command "n8n start"

# ── Done ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  All services started!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""
Write-Host "  API        -> http://localhost:3000/api/v1" -ForegroundColor White
Write-Host "  Swagger    -> http://localhost:3000/api/docs" -ForegroundColor White
Write-Host "  n8n        -> http://localhost:5678" -ForegroundColor White
Write-Host "  MinIO      -> http://localhost:9001" -ForegroundColor White
Write-Host "  RabbitMQ   -> http://localhost:15672  (guest/guest)" -ForegroundColor White
Write-Host ""
Write-Host "  Anomaly    -> http://localhost:8001/health" -ForegroundColor Gray
Write-Host "  Classify   -> http://localhost:8002/health" -ForegroundColor Gray
Write-Host "  NLP        -> http://localhost:8003/health" -ForegroundColor Gray
Write-Host "  Vision     -> http://localhost:8004/health" -ForegroundColor Gray
Write-Host ""
Write-Host "  Wait ~15 seconds for all services to be ready." -ForegroundColor Yellow
Write-Host ""

# Keep this window open
Read-Host "Press Enter to close this launcher"
