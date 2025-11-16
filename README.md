# Document Parser - Production-Grade PDF to Markdown Converter

High-performance document processing system with hybrid OCR, intelligent page routing, and async job queue for production deployments.

## 🚀 Overview

A production-ready document parser that combines multiple OCR models (DeepSeek-OCR, Nanonets-OCR) with intelligent routing, async processing, and Redis-backed job queue for handling concurrent users and large documents without timeouts.

### Key Features

- **Hybrid Multi-Model OCR**: Automatic routing between DeepSeek and Nanonets models
- **Async Job Queue**: Redis + RQ for background processing, no timeouts
- **Concurrent Processing**: Multiple workers handle parallel jobs
- **Fast PDF Conversion**: PyMuPDF (3-5x faster than pdf2image)
- **Production Ready**: Auto-scaling workers, health checks, OpenAPI docs
- **Real-time Progress**: Track processing status per page
- **Upload Progress**: See file upload progress in real-time

---

## 📐 Architecture

### System Overview

```
┌─────────────┐
│   Client    │
│ (Browser)   │
└──────┬──────┘
       │ HTTP
       ▼
┌─────────────────────────────────────────────┐
│           FastAPI Server (Port 1233)        │
│  ┌────────────┐  ┌──────────────────────┐  │
│  │ Sync API   │  │   Async Job Queue    │  │
│  │ /v1/process│  │  /v1/process/async   │  │
│  │   /sync    │  │  /v1/jobs/{id}/*     │  │
│  └────────────┘  └──────────────────────┘  │
└─────────┬───────────────────┬───────────────┘
          │                   │
          │                   ▼
          │          ┌─────────────────┐
          │          │  Redis (6379)   │
          │          │  Job Queue +    │
          │          │  State Storage  │
          │          └────────┬────────┘
          │                   │
          │                   ▼
          │          ┌─────────────────┐
          │          │  RQ Workers (2) │
          │          │  Background     │
          │          │  Processing     │
          │          └────────┬────────┘
          │                   │
          ▼                   ▼
┌────────────────────────────────────────────┐
│      Hybrid Document Processor             │
│  ┌──────────────────────────────────────┐ │
│  │  AsyncVlmProcessor (concurrency=16)  │ │
│  │  ┌──────────┐      ┌──────────┐     │ │
│  │  │ DeepSeek │      │ Nanonets │     │ │
│  │  │  Router  │      │  Router  │     │ │
│  │  └────┬─────┘      └────┬─────┘     │ │
│  └───────┼──────────────────┼───────────┘ │
└──────────┼──────────────────┼─────────────┘
           │                  │
           ▼                  ▼
    ┌─────────────┐    ┌─────────────┐
    │  vLLM:4444  │    │  vLLM:4445  │
    │  DeepSeek   │    │  Nanonets   │
    │  OCR Model  │    │  OCR Model  │
    └─────────────┘    └─────────────┘
```

### Components

#### 1. **FastAPI Server** (`api.py`)
- REST API with OpenAPI/Swagger docs
- Handles file uploads and job management
- Auto-starts background workers on startup
- CORS enabled for web clients

#### 2. **Job Queue System**
- **Redis**: In-memory job queue and state storage
- **RQ (Redis Queue)**: Simple Python job queue
- **Workers**: 2 parallel workers (configurable)
- **TTL**: 24-hour result retention

#### 3. **Hybrid Document Processor** (`hybrid_processor.py`)
- Orchestrates multi-model processing
- Intelligent page routing (table detection)
- Combines results into unified output

#### 4. **Async VLM Processor** (`async_processor.py`)
- Parallel page processing (16 concurrent)
- PyMuPDF for fast PDF→Image conversion
- Per-page model routing
- Async API calls to vLLM servers

#### 5. **External vLLM Servers**
- **DeepSeek-OCR** (Port 4444): Text-heavy pages
- **Nanonets-OCR** (Port 4445): Tables and complex layouts
- `max-num-seqs` for concurrent request handling

---

## 🔧 Installation

### Prerequisites

- **Python**: 3.12+
- **Redis**: Running on localhost:6379
- **vLLM Servers**: DeepSeek-OCR (4444), Nanonets-OCR (4445)

### Setup

```bash
# Clone repository
git clone <repo-url>
cd document-parser

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### Configuration (`.env`)

```bash
# =============================================================================
# API Server
# =============================================================================
API_HOST=0.0.0.0
API_PORT=1233

# =============================================================================
# External vLLM Servers (OpenAI-compatible)
# =============================================================================
# Note: AsyncOpenAI auto-appends /chat/completions, only provide base URL
VLLM_DEEPSEEK_URL=http://localhost:4444/v1
VLLM_NANONETS_URL=http://localhost:4445/v1

# =============================================================================
# Redis Configuration
# =============================================================================
REDIS_URL=redis://localhost:6379/0

# =============================================================================
# Background Workers
# =============================================================================
NUM_WORKERS=2  # Number of concurrent job processors

# =============================================================================
# Processing Settings
# =============================================================================
DEFAULT_ACCURACY_MODE=balanced  # fast | balanced | maximum
INFERENCE_MODE=vllm             # vllm (external servers)
ENABLE_ENRICHMENT=false         # Semantic enrichment (optional)
```

---

## 🚦 Starting the Server

### Option 1: Quick Start (Recommended)

```bash
./start.sh
```

**What it does:**
- ✅ Checks Redis connectivity
- ✅ Verifies vLLM servers
- ✅ Validates dependencies
- ✅ Starts API + auto-starts 2 workers
- ✅ Graceful shutdown on Ctrl+C

### Option 2: Manual Start

```bash
# Start API (workers auto-start)
python -m document_parser.api
```

---

## 📡 API Documentation

### Base URL

- **Local**: `http://localhost:1233`
- **Production**: `https://lmstudio.subh-dev.xyz`

### Interactive Docs

- **Swagger UI**: `http://localhost:1233/docs`
- **ReDoc**: `http://localhost:1233/redoc`
- **OpenAPI Spec**: `http://localhost:1233/openapi.json`

---

## 🛣️ API Routes

### Info & Health

#### `GET /`
Root endpoint with service info.

**Response:**
```json
{
  "service": "Hybrid Document Parser API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

#### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "processors": {
    "deepseek": true,
    "nanonets": true
  },
  "config": {
    "accuracy_mode": "balanced",
    "inference_mode": "vllm",
    "device": "cuda"
  }
}
```

---

### Document Processing

#### `POST /v1/process/sync`
Process document synchronously (returns immediately with result).

**Use for:** Small documents (<50 pages), when you need immediate results.

**Request:**
```bash
curl -X POST "http://localhost:1233/v1/process/sync" \
  -F "file=@document.pdf" \
  -F "accuracy_mode=balanced" \
  -F "output_format=markdown"
```

**Parameters:**
- `file` (required): PDF file
- `accuracy_mode` (optional): `fast` | `balanced` | `maximum` (default: balanced)
- `output_format` (optional): `markdown` | `json` | `html` (default: json)
- `generate_embeddings` (optional): boolean (default: false)

**Response:**
```json
{
  "success": true,
  "document_id": "abc123",
  "filename": "document.pdf",
  "nodes": [...],
  "edges": [...],
  "metadata": {
    "num_pages": 26,
    "processing_time": 35.4,
    "deepseek_pages": 10,
    "nanonets_pages": 16,
    "output": "# Markdown content here..."
  }
}
```

#### `POST /v1/process/async`
Submit document for async background processing (recommended for production).

**Use for:** Large documents, multiple concurrent users, avoiding timeouts.

**Request:**
```bash
curl -X POST "http://localhost:1233/v1/process/async" \
  -F "file=@large_document.pdf" \
  -F "accuracy_mode=balanced" \
  -F "output_format=markdown"
```

**Response (202 Accepted):**
```json
{
  "job_id": "feb8c239-b1ad-4020-beb5-eed33897b64f",
  "status": "queued",
  "message": "Job submitted successfully",
  "status_url": "/v1/jobs/feb8c239-b1ad-4020-beb5-eed33897b64f/status",
  "result_url": "/v1/jobs/feb8c239-b1ad-4020-beb5-eed33897b64f/result"
}
```

---

### Job Management

#### `GET /v1/jobs/{job_id}/status`
Check job status and progress.

**Request:**
```bash
curl "http://localhost:1233/v1/jobs/feb8c239-b1ad-4020-beb5-eed33897b64f/status"
```

**Response (Queued):**
```json
{
  "job_id": "feb8c239-b1ad-4020-beb5-eed33897b64f",
  "status": "queued",
  "filename": "document.pdf",
  "created_at": "2025-11-16T15:25:49.807Z",
  "progress": {}
}
```

**Response (Processing):**
```json
{
  "job_id": "feb8c239-b1ad-4020-beb5-eed33897b64f",
  "status": "processing",
  "filename": "document.pdf",
  "created_at": "2025-11-16T15:25:49.807Z",
  "started_at": "2025-11-16T15:25:50.100Z",
  "progress": {
    "current_page": 127,
    "total_pages": 207,
    "deepseek_pages": 89,
    "nanonets_pages": 38
  }
}
```

**Response (Completed):**
```json
{
  "job_id": "feb8c239-b1ad-4020-beb5-eed33897b64f",
  "status": "completed",
  "filename": "document.pdf",
  "created_at": "2025-11-16T15:25:49.807Z",
  "started_at": "2025-11-16T15:25:50.100Z",
  "completed_at": "2025-11-16T15:28:12.500Z",
  "progress": {
    "total_pages": 207,
    "processing_time": 122.8,
    "deepseek_pages": 123,
    "nanonets_pages": 84
  }
}
```

**Response (Failed):**
```json
{
  "job_id": "feb8c239-b1ad-4020-beb5-eed33897b64f",
  "status": "failed",
  "filename": "document.pdf",
  "error": "vLLM server timeout"
}
```

#### `GET /v1/jobs/{job_id}/result`
Download completed job result.

**Request:**
```bash
curl "http://localhost:1233/v1/jobs/feb8c239-b1ad-4020-beb5-eed33897b64f/result" \
  -o result.md
```

**Response:**
```json
{
  "markdown": "# Document Title\n\n## Section 1\n...",
  "filename": "document.pdf"
}
```

**Errors:**
- `404`: Job not found
- `400`: Job not completed yet (status: queued/processing)

#### `GET /v1/jobs`
List recent jobs.

**Request:**
```bash
curl "http://localhost:1233/v1/jobs?limit=10"
```

**Parameters:**
- `limit` (optional): Max results (default: 100, max: 1000)

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "...",
      "status": "completed",
      "filename": "doc1.pdf",
      "created_at": "..."
    },
    {
      "job_id": "...",
      "status": "processing",
      "filename": "doc2.pdf",
      "created_at": "..."
    }
  ],
  "total": 2
}
```

#### `DELETE /v1/jobs/{job_id}`
Delete job and its result.

**Request:**
```bash
curl -X DELETE "http://localhost:1233/v1/jobs/feb8c239-b1ad-4020-beb5-eed33897b64f"
```

**Response:**
```json
{
  "message": "Job deleted successfully"
}
```

---

### Capabilities

#### `GET /v1/capabilities`
Get processor model capabilities.

**Response:**
```json
{
  "deepseek": {
    "model": "DeepSeek-OCR",
    "supports": ["text", "math", "code"],
    "max_tokens": 4096
  },
  "nanonets": {
    "model": "Nanonets-OCR2-3B",
    "supports": ["tables", "forms", "structured_data"],
    "max_tokens": 4096
  }
}
```

---

## 📊 Usage Examples

### Web Interface

Open `upload.html` in a browser or serve it:

```bash
# Navigate to upload.html in browser
open upload.html
```

**Features:**
- Drag & drop PDF upload
- Real-time upload progress
- Processing status updates
- Download markdown result

### Python Client

```python
import requests
import time

API_BASE = "http://localhost:1233"

# Submit async job
with open("document.pdf", "rb") as f:
    response = requests.post(
        f"{API_BASE}/v1/process/async",
        files={"file": f},
        data={
            "accuracy_mode": "balanced",
            "output_format": "markdown"
        }
    )

job = response.json()
job_id = job["job_id"]
print(f"Job submitted: {job_id}")

# Poll status
while True:
    status = requests.get(f"{API_BASE}/v1/jobs/{job_id}/status").json()

    if status["status"] == "completed":
        print(f"✓ Completed in {status['progress']['processing_time']:.1f}s")
        break
    elif status["status"] == "failed":
        print(f"✗ Failed: {status['error']}")
        break
    elif status["status"] == "processing":
        prog = status["progress"]
        print(f"Processing: {prog.get('current_page', 0)}/{prog.get('total_pages', 0)}")

    time.sleep(2)

# Download result
result = requests.get(f"{API_BASE}/v1/jobs/{job_id}/result").json()
markdown = result["markdown"]

with open("output.md", "w") as f:
    f.write(markdown)
```

### cURL Examples

**Quick sync processing:**
```bash
curl -X POST "http://localhost:1233/v1/process/sync" \
  -F "file=@test.pdf" \
  -F "output_format=markdown" \
  | jq -r '.metadata.output' > output.md
```

**Async with polling:**
```bash
# Submit
JOB_ID=$(curl -X POST "http://localhost:1233/v1/process/async" \
  -F "file=@large.pdf" \
  | jq -r '.job_id')

# Check status
curl "http://localhost:1233/v1/jobs/$JOB_ID/status" | jq

# Download result
curl "http://localhost:1233/v1/jobs/$JOB_ID/result" \
  | jq -r '.markdown' > output.md
```

---

## ⚡ Performance

### Throughput

| Accuracy Mode | Pages/Day (1 worker) | Pages/Day (2 workers) |
|---------------|----------------------|----------------------|
| Fast          | 5,000-8,000         | 10,000-16,000        |
| Balanced      | 2,000-3,000         | 4,000-6,000          |
| Maximum       | 1,000-1,500         | 2,000-3,000          |

### Concurrency Levels

1. **Job-level**: `NUM_WORKERS=2` (2 PDFs processed simultaneously)
2. **Page-level**: `concurrency=16` (16 pages per PDF in parallel)
3. **vLLM-level**:
   - DeepSeek: `--max-num-seqs 32`
   - Nanonets: `--max-num-seqs 8`

### Example Processing Times

- **26-page PDF**: ~35s (0.73 pages/sec)
- **207-page PDF**: ~123s (1.68 pages/sec)
- **371-page PDF**: ~220s (1.69 pages/sec)

---

## 🔐 Production Deployment

### Scaling Workers

```bash
# .env
NUM_WORKERS=4  # 4 concurrent jobs

# Or environment variable
NUM_WORKERS=8 ./start.sh
```

**Recommendation**: 1-2 workers per CPU core

### Reverse Proxy (Nginx)

```nginx
upstream document_parser {
    server localhost:1233;
}

server {
    listen 80;
    server_name api.example.com;

    client_max_body_size 100M;  # Allow large PDFs

    location / {
        proxy_pass http://document_parser;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 600s;  # Timeout for sync endpoint
    }
}
```

### Systemd Service

```ini
# /etc/systemd/system/document-parser.service
[Unit]
Description=Document Parser API
After=network.target redis.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/document-parser
Environment="PATH=/opt/document-parser/.venv/bin"
ExecStart=/opt/document-parser/.venv/bin/python -m document_parser.api
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable document-parser
sudo systemctl start document-parser
sudo systemctl status document-parser
```

### Monitoring

**Check worker status:**
```bash
# Redis CLI
redis-cli

# List jobs
KEYS job:*

# View job data
GET job:feb8c239-b1ad-4020-beb5-eed33897b64f

# Queue length
LLEN rq:queue:default
```

**API metrics:**
```bash
# Health check
curl http://localhost:1233/health | jq

# List active jobs
curl http://localhost:1233/v1/jobs | jq
```

---

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Find process using port
lsof -i :1233

# Kill process
kill <PID>

# Or change port in .env
API_PORT=8080
```

### Redis Connection Failed

```bash
# Check Redis
redis-cli ping

# Start Redis
sudo systemctl start redis

# Or
redis-server &
```

### vLLM Server Not Responding

```bash
# Check DeepSeek server
curl http://localhost:4444/health

# Check Nanonets server
curl http://localhost:4445/health

# Restart vLLM servers
# (see vLLM documentation)
```

### Worker Not Processing Jobs

```bash
# Check worker logs
journalctl -u document-parser -f

# Check Redis queue
redis-cli LLEN rq:queue:default

# Check if workers are running
ps aux | grep python | grep worker
```

### Upload Timeout/Slow

**Issue**: Large PDFs timing out or uploading slowly

**Cause**: Network bandwidth limitation (not an API issue)

**Example**: 50 MB PDF on 10 Mbps upload = ~40 seconds

**Solution**: This is normal. The upload progress indicator in the web UI shows real-time progress.

---

## 📁 Project Structure

```
document-parser/
├── document_parser/
│   ├── __init__.py
│   ├── api.py                  # FastAPI server + endpoints
│   ├── worker.py               # RQ background worker
│   ├── job_manager.py          # Redis job state management
│   ├── hybrid_processor.py    # Multi-model orchestrator
│   ├── async_processor.py     # Async page processor
│   ├── config.py               # Settings & configuration
│   ├── deepseek_ocr.py         # DeepSeek model client
│   └── nanonets_ocr.py         # Nanonets model client
│
├── uploads/                    # Temporary PDF storage
├── outputs/                    # Processed results (sync mode)
├── .env                        # Configuration (create from .env.example)
├── .env.example                # Configuration template
├── pyproject.toml              # Dependencies
├── start.sh                    # Production startup script
├── upload.html                 # Web UI
└── README.md                   # This file
```

---

## 📝 What This Does NOT Do

- ❌ Deploy vLLM servers (they run externally)
- ❌ Load AI models into GPU (models run on your vLLM servers)
- ❌ Require GPU access (API server is CPU-only)

---

## 📞 Support

- **Interactive Docs**: `http://localhost:1233/docs`
- **OpenAPI Spec**: `http://localhost:1233/openapi.json`
- **Health Check**: `http://localhost:1233/health`
