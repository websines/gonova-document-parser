# Production Deployment Guide

## Overview

The document parser uses a **production-grade async job queue** system with:

- **Redis** for job queue and state persistence
- **RQ (Redis Queue)** for background processing
- **Multiple workers** for parallel processing
- **No timeouts** - handles large PDFs gracefully

## Architecture

```
User uploads PDF
    ↓
API returns job_id (202 Accepted)
    ↓
Job queued in Redis
    ↓
Worker processes job in background
    ↓
User polls /v1/jobs/{job_id}/status
    ↓
User downloads result when complete
```

## Prerequisites

1. **Redis** running on localhost:6379
2. **vLLM servers** running:
   - Port 4444: DeepSeek-OCR
   - Port 4445: Nanonets-OCR2-3B
3. **Python 3.10+** with dependencies installed

## Installation

```bash
# Install dependencies
pip install -e .

# Or install with all optional dependencies
pip install -e ".[api,db,dev]"
```

## Running in Production

### One-Command Startup (Recommended)

The API automatically starts background workers - **just one command!**

```bash
# Development (with auto-reload)
python -m document_parser.api

# Production (with Gunicorn)
gunicorn document_parser.api:app \
    --workers 4 \
    --bind 0.0.0.0:1233 \
    --timeout 300 \
    --worker-class uvicorn.workers.UvicornWorker
```

**Background workers start automatically!**
- Default: 2 workers
- Configure with `NUM_WORKERS=4` in .env
- Workers auto-terminate when API stops

### Alternative: Separate Workers (For Advanced Deployments)

If you want workers on different servers:

```bash
# Terminal 1: API only (set NUM_WORKERS=0 in .env)
python -m document_parser.api

# Terminal 2+: Manual workers on different machines
python -m document_parser.worker
```

### 3. Monitor Jobs (Optional)

```bash
# Start RQ Dashboard for monitoring
rq-dashboard --redis-url redis://localhost:6379
# Access at http://localhost:9181
```

## API Endpoints

### Async Processing (Recommended for Production)

**Submit job:**
```bash
curl -X POST "https://your-domain.com/v1/process/async" \
  -F "file=@document.pdf" \
  -F "accuracy_mode=balanced" \
  -F "output_format=markdown"

# Response:
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Job submitted successfully",
  "status_url": "/v1/jobs/550e8400-e29b-41d4-a716-446655440000/status",
  "result_url": "/v1/jobs/550e8400-e29b-41d4-a716-446655440000/result"
}
```

**Check status:**
```bash
curl "https://your-domain.com/v1/jobs/550e8400-e29b-41d4-a716-446655440000/status"

# Response:
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",  # or "queued", "completed", "failed"
  "filename": "document.pdf",
  "progress": {
    "current_page": 50,
    "total_pages": 100,
    "deepseek_pages": 70,
    "nanonets_pages": 30
  },
  "created_at": "2025-11-16T14:30:00",
  "started_at": "2025-11-16T14:30:05"
}
```

**Download result:**
```bash
curl "https://your-domain.com/v1/jobs/550e8400-e29b-41d4-a716-446655440000/result" \
  -o result.md

# Response:
{
  "markdown": "# Document Content\n\n...",
  "filename": "document.pdf"
}
```

**List jobs:**
```bash
curl "https://your-domain.com/v1/jobs?limit=50"
```

### Sync Processing (For Small Documents Only)

```bash
curl -X POST "https://your-domain.com/v1/process/sync" \
  -F "file=@small-doc.pdf" \
  -F "accuracy_mode=balanced" \
  -F "output_format=markdown"
```

⚠️ **Warning:** Sync endpoint may timeout on large documents (>100 pages) or under heavy load. Use async endpoint for production.

## Configuration

### .env File

```bash
# Redis (Required for async jobs)
REDIS_URL=redis://localhost:6379/0

# vLLM Servers (Required)
VLLM_DEEPSEEK_URL=http://localhost:4444/v1
VLLM_NANONETS_URL=http://localhost:4445/v1

# Processing
DEFAULT_ACCURACY_MODE=balanced
INFERENCE_MODE=vllm

# API Server
API_HOST=0.0.0.0
API_PORT=1233
```

## Systemd Service (Linux)

Create `/etc/systemd/system/docparse-api.service`:

```ini
[Unit]
Description=Document Parser API
After=network.target redis.service

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/document-parser
Environment=PATH=/path/to/venv/bin
ExecStart=/path/to/venv/bin/gunicorn document_parser.api:app \
    --workers 4 \
    --bind 0.0.0.0:1233 \
    --timeout 300 \
    --worker-class uvicorn.workers.UvicornWorker
Restart=always

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/docparse-worker@.service`:

```ini
[Unit]
Description=Document Parser Worker %i
After=network.target redis.service

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/document-parser
Environment=PATH=/path/to/venv/bin
ExecStart=/path/to/venv/bin/python -m document_parser.worker
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
# Start API
sudo systemctl enable --now docparse-api

# Start 3 workers
sudo systemctl enable --now docparse-worker@1
sudo systemctl enable --now docparse-worker@2
sudo systemctl enable --now docparse-worker@3
```

## Performance

**Async Job Queue Benefits:**
- ✅ No timeouts - processes huge PDFs (1000+ pages)
- ✅ Multiple concurrent users - jobs queue automatically
- ✅ Progress tracking - see real-time page counts
- ✅ State persistence - survives server restarts
- ✅ Auto-cleanup - old jobs expire after 24 hours
- ✅ Multiple workers - true parallel processing

**Expected Performance:**
- Small PDF (10 pages): ~3-5 seconds
- Medium PDF (100 pages): ~30-60 seconds
- Large PDF (371 pages): ~2-4 minutes
- Very large PDF (1000 pages): ~8-12 minutes

## Monitoring

### Check Worker Status
```bash
# List running workers
rq info --url redis://localhost:6379

# Watch queue in real-time
rq info --url redis://localhost:6379 --interval 1
```

### Check Redis
```bash
# Connect to Redis
redis-cli

# List all jobs
KEYS job:*:status

# Get job details
GET job:550e8400-e29b-41d4-a716-446655440000:status
```

## Troubleshooting

### Jobs stuck in "queued" status
- Check if workers are running: `ps aux | grep worker`
- Check worker logs
- Restart workers

### Redis connection errors
- Verify Redis is running: `redis-cli ping`
- Check REDIS_URL in .env

### Out of memory errors
- Reduce number of workers
- Add more RAM
- Enable Redis persistence: `redis-cli CONFIG SET maxmemory-policy allkeys-lru`

## Scaling

**Horizontal Scaling (Multiple Servers):**
1. Run API servers on multiple machines (all connect to same Redis)
2. Run workers on multiple machines (all connect to same Redis)
3. Load balance API servers with nginx/haproxy
4. Redis cluster for high availability

**Vertical Scaling (Single Server):**
1. Add more workers (1-2 per CPU core)
2. Increase Redis max memory
3. Use SSD for faster I/O

## Security

- ✅ Rate limiting (configure in API)
- ✅ File size limits (configure in API)
- ✅ CORS configuration (update in api.py)
- ✅ Redis password (set in REDIS_URL)
- ✅ HTTPS/TLS (configure reverse proxy)

## Maintenance

**Cleanup old jobs manually:**
```python
from document_parser.job_manager import JobManager
manager = JobManager()
manager.cleanup_old_jobs(days=7)  # Delete jobs older than 7 days
```

**Backup Redis data:**
```bash
# Enable RDB snapshots
redis-cli CONFIG SET save "900 1 300 10 60 10000"

# Manual backup
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb /backup/redis-backup-$(date +%Y%m%d).rdb
```
