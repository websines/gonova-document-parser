# vLLM Deployment Guide for WSL + NVIDIA 3090

Complete guide for deploying Document Parser with vLLM on your bare metal PC.

## Why vLLM?

- **2-3x faster** than Transformers mode
- **Better throughput**: Dynamic batching, continuous batching
- **OpenAI-compatible API**: Easy integration
- **Production-ready**: Handles 3000-5000 pages/day on 3090

## Prerequisites

✅ Windows 11 with WSL2
✅ NVIDIA 3090 (24GB VRAM)
✅ CUDA 12.1+ accessible in WSL
✅ Ubuntu/Debian in WSL

## Quick Start (5 Minutes)

```bash
# 1. Setup vLLM
cd /path/to/document-parser
chmod +x scripts/*.sh
./scripts/setup_vllm_wsl.sh

# 2. Start everything
./scripts/quickstart_vllm.sh

# 3. Test it
curl http://localhost:8080/health
```

Done! API is at `http://localhost:8080/docs`

## Detailed Setup

### Step 1: Install vLLM

```bash
# Activate venv
cd document-parser
source .venv/bin/activate

# Install vLLM (takes 5-10 min)
pip install vllm

# Verify
python -c "import vllm; print(f'vLLM {vllm.__version__}')"
```

### Step 2: Pre-download Models (Recommended)

```bash
# Download all models (~10GB total)
python << 'EOF'
from huggingface_hub import snapshot_download

models = [
    "deepseek-ai/DeepSeek-OCR",           # 3B - primary
    "nanonets/Nanonets-OCR2-3B",          # 4B - handwriting
    "Qwen/Qwen3-Embedding-0.6B",          # 0.6B - embeddings
]

for model in models:
    print(f"Downloading {model}...")
    snapshot_download(model)
EOF
```

### Step 3: Configure Environment

```bash
# Copy and edit .env
cp .env.example .env
nano .env
```

**Critical settings for vLLM**:
```bash
# Inference Mode
INFERENCE_MODE=vllm

# vLLM Endpoints
VLLM_DEEPSEEK_URL=http://localhost:8000/v1
VLLM_NANONETS_URL=http://localhost:8001/v1
VLLM_GRANITE_URL=http://localhost:8002/v1

# GPU
CUDA_VISIBLE_DEVICES=0
VRAM_LIMIT_GB=22

# Redis
REDIS_URL=redis://localhost:6379/0
```

### Step 4: Start Services

**Option A: Interactive (Recommended for testing)**

```bash
# Terminal 1: vLLM Server
./scripts/vllm_manager.sh start deepseek

# Terminal 2: API Server
uvicorn document_parser.api:app --host 0.0.0.0 --port 8080

# Terminal 3: Monitor
watch -n 2 nvidia-smi
```

**Option B: Quick Start Script**

```bash
# Starts Redis + vLLM + API in one go
./scripts/quickstart_vllm.sh
```

**Option C: Background Services**

```bash
# Start vLLM in background
./scripts/vllm_manager.sh start deepseek

# Start API in background
nohup uvicorn document_parser.api:app --host 0.0.0.0 --port 8080 > logs/api.log 2>&1 &
```

## Model Management on 3090

The 3090 has 24GB VRAM. You can run ONE primary model at a time:

### Strategy: Primary + Swap

Run **DeepSeek** (handles 80% of documents), swap to **Nanonets** when needed:

```bash
# Start with DeepSeek (primary)
./scripts/vllm_manager.sh start deepseek

# Later, swap to Nanonets for handwriting docs
./scripts/vllm_manager.sh swap nanonets

# Swap back to DeepSeek
./scripts/vllm_manager.sh swap deepseek

# Check what's running
./scripts/vllm_manager.sh status
```

### Model Specifications

| Model | VRAM | Speed | Use Case |
|-------|------|-------|----------|
| **DeepSeek** | ~18GB | Fast | Standard documents (80%) |
| **Nanonets** | ~20GB | Medium | Handwriting, signatures (15%) |
| **Granite** | ~2GB | Very fast | Enrichment only (5%) |

### VRAM Breakdown

```
GPU Memory (24GB total):
├─ System overhead: ~2GB
├─ vLLM overhead: ~2GB
├─ Model weights: 12-18GB (depends on model)
├─ KV cache: 2-4GB (dynamic)
└─ Available: ~22GB usable
```

## vLLM Manager Commands

```bash
# Start a model
./scripts/vllm_manager.sh start deepseek
./scripts/vllm_manager.sh start nanonets

# Stop models
./scripts/vllm_manager.sh stop              # Stop all
./scripts/vllm_manager.sh stop deepseek     # Stop specific

# Restart
./scripts/vllm_manager.sh restart deepseek

# Swap (stops others, starts target)
./scripts/vllm_manager.sh swap nanonets

# Check status
./scripts/vllm_manager.sh status

# View logs
./scripts/vllm_manager.sh logs deepseek
```

## Production Deployment (Auto-start on Boot)

### Install Systemd Services

```bash
# 1. Edit service files with your paths
nano scripts/systemd/docparser-vllm.service
nano scripts/systemd/docparser-api.service

# Replace:
#   YOUR_USERNAME → your actual username
#   /path/to/document-parser → actual path

# 2. Install services
sudo cp scripts/systemd/*.service /etc/systemd/system/

# 3. Enable auto-start
sudo systemctl daemon-reload
sudo systemctl enable docparser-vllm
sudo systemctl enable docparser-api

# 4. Start now
sudo systemctl start docparser-vllm
sudo systemctl start docparser-api

# 5. Check status
sudo systemctl status docparser-vllm
sudo systemctl status docparser-api
```

### View Logs

```bash
# Service logs
sudo journalctl -u docparser-vllm -f
sudo journalctl -u docparser-api -f

# Or direct log files
tail -f logs/vllm_deepseek.log
tail -f logs/api.log
```

### Manage Services

```bash
# Start/Stop
sudo systemctl start docparser-api
sudo systemctl stop docparser-api

# Restart
sudo systemctl restart docparser-vllm

# Disable auto-start
sudo systemctl disable docparser-api
```

## Performance Tuning

### 1. Optimal vLLM Settings for 3090

```bash
vllm serve MODEL \
  --gpu-memory-utilization 0.85 \     # Use 85% of 24GB
  --max-model-len 8192 \              # Context length
  --max-num-seqs 8 \                  # Batch size (auto-tuned)
  --dtype bfloat16 \                  # Memory efficient
  --trust-remote-code                 # Required for custom models
```

### 2. Monitor Performance

```bash
# GPU utilization
watch -n 1 nvidia-smi

# API stats
curl http://localhost:8080/v1/jobs | jq '.stats'

# vLLM metrics (if enabled)
curl http://localhost:8000/metrics
```

### 3. Benchmarking

```bash
# Process 10 documents and measure
time for i in {1..10}; do
  curl -X POST "http://localhost:8080/v1/process" \
    -F "file=@test.pdf" \
    -F "accuracy_mode=balanced"
  sleep 1
done
```

## Troubleshooting

### vLLM won't start

```bash
# Check CUDA
nvidia-smi

# Check if port is in use
lsof -i :8000

# View logs
tail -f logs/vllm_deepseek.log

# Try manual start (see errors)
vllm serve deepseek-ai/DeepSeek-OCR --port 8000
```

### Out of Memory

```bash
# Reduce GPU memory utilization
--gpu-memory-utilization 0.75  # Instead of 0.85

# Reduce context length
--max-model-len 4096  # Instead of 8192

# Check what's using GPU
nvidia-smi

# Kill other processes
pkill -f vllm
```

### Slow Performance

```bash
# Check GPU utilization (should be 80-100%)
nvidia-smi

# Check if using GPU
curl http://localhost:8000/v1/models | jq '.data[0]'

# Increase batch size
--max-num-seqs 16  # Instead of 8

# Check logs for bottlenecks
tail -f logs/api.log
```

### API Connection Issues

```bash
# Check if vLLM is running
./scripts/vllm_manager.sh status

# Test vLLM directly
curl http://localhost:8000/v1/models

# Check API can reach vLLM
curl http://localhost:8080/health | jq '.processors'

# Verify .env settings
cat .env | grep VLLM_
```

## Access from Windows/Network

### From Windows Host

The API runs in WSL, accessible from Windows via `localhost`:

```powershell
# From Windows PowerShell
curl http://localhost:8080/health
```

### From Other Devices (LAN)

```bash
# Get WSL IP
ip addr show eth0 | grep "inet " | awk '{print $2}' | cut -d/ -f1

# Access from other devices
http://WSL_IP:8080
```

**Port forwarding** (if needed):
```powershell
# Run as Administrator in Windows PowerShell
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=WSL_IP

# List port forwards
netsh interface portproxy show all

# Remove
netsh interface portproxy delete v4tov4 listenport=8080 listenaddress=0.0.0.0
```

## Daily Operations

### Morning Startup

```bash
# Check status
./scripts/vllm_manager.sh status

# If not running, start everything
./scripts/quickstart_vllm.sh
```

### During the Day

```bash
# Monitor GPU
watch -n 5 nvidia-smi

# Check queue
curl http://localhost:8080/v1/jobs | jq '.stats'

# Swap models if needed
./scripts/vllm_manager.sh swap nanonets
```

### Evening Shutdown

```bash
# Check remaining jobs
curl http://localhost:8080/v1/jobs?status=processing

# Stop services (optional - can run 24/7)
./scripts/vllm_manager.sh stop
pkill -f uvicorn
```

## Performance Expectations on 3090

### Transformers Mode (Baseline)
- **Speed**: 0.8-1.2 pages/sec
- **Daily**: 2000-3000 pages/day (balanced mode)
- **Latency**: ~1-2 seconds per page

### vLLM Mode (Production)
- **Speed**: 1.5-2.5 pages/sec ⚡
- **Daily**: 3000-5000 pages/day (balanced mode) ⚡
- **Latency**: ~0.5-1 second per page ⚡
- **Speedup**: **2-3x faster**

### With Embeddings
- **Overhead**: +2-3 seconds per document (50-100 nodes)
- **Speed**: 1.2-2.0 pages/sec
- **Daily**: 2500-4000 pages/day

## Next Steps

1. ✅ Run setup: `./scripts/setup_vllm_wsl.sh`
2. ✅ Start services: `./scripts/quickstart_vllm.sh`
3. ✅ Test API: `curl http://localhost:8080/docs`
4. ✅ Process first document via Swagger UI
5. ✅ Set up systemd for auto-start (optional)
6. ✅ Monitor performance for 1-2 days
7. ✅ Tune settings based on your workload

## Support

- **Logs**: `tail -f logs/*.log`
- **Status**: `./scripts/vllm_manager.sh status`
- **GPU**: `nvidia-smi`
- **API Health**: `curl http://localhost:8080/health`
