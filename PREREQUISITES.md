# Prerequisites

This application **connects to external services** - it does not deploy them.

## Required Services (Must Be Running)

### 1. **vLLM Servers** (3 separate instances)

You must have these running on your host machine:

```bash
# Port 4444 - DeepSeek-OCR
vllm serve deepseek-ai/DeepSeek-OCR \
  --host 0.0.0.0 \
  --port 4444 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --trust-remote-code

# Port 4445 - Nanonets-OCR2-3B
vllm serve nanonets/Nanonets-OCR2-3B \
  --host 0.0.0.0 \
  --port 4445 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 15000 \
  --trust-remote-code

# Port 4446 - Granite-Docling-258M (optional, for enrichment)
vllm serve ibm-granite/granite-docling-258m \
  --host 0.0.0.0 \
  --port 4446 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.5 \
  --max-model-len 8192 \
  --trust-remote-code
```

**Note:** On NVIDIA 3090 (24GB VRAM), you can typically run **one** vLLM server at a time. Swap models as needed.

### 2. **Redis Server**

Required for job queue persistence:

```bash
# Using Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or native Redis
redis-server
```

Verify it's running:
```bash
redis-cli ping  # Should return "PONG"
```

## Verification Checklist

Before starting this application, verify all services are accessible:

```bash
# Check vLLM servers
curl http://localhost:4444/v1/models  # DeepSeek
curl http://localhost:4445/v1/models  # Nanonets
curl http://localhost:4446/v1/models  # Granite (optional)

# Check Redis
redis-cli ping  # Should return "PONG"
```

## What This Application Does

This application **only**:
- ✅ Accepts PDF uploads via REST API
- ✅ Analyzes PDFs and routes to appropriate vLLM server
- ✅ Manages job queue with Redis
- ✅ Returns processed markdown/JSON results

This application **does NOT**:
- ❌ Deploy vLLM servers
- ❌ Load AI models into GPU
- ❌ Start Redis
- ❌ Require GPU access (all GPU work happens in external vLLM servers)

## Quick Start

Once all prerequisites are running:

```bash
# Copy environment template
cp .env.example .env

# Verify external services (should all respond)
curl http://localhost:4444/v1/models
redis-cli ping

# Start the API server
./start_api.sh

# Or manually:
uvicorn document_parser.api:app --host 0.0.0.0 --port 8080
```

API will be available at:
- Main API: http://localhost:8080
- Swagger Docs: http://localhost:8080/docs
- Health Check: http://localhost:8080/health
