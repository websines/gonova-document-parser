#!/bin/bash
# Start vLLM servers for document processing on NVIDIA 3090
# Each model runs on a separate port with optimized configuration

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting vLLM servers for document processing...${NC}"
echo ""

# Check if vLLM is installed
if ! command -v vllm &> /dev/null; then
    echo -e "${YELLOW}vLLM not found. Install with: pip install vllm${NC}"
    exit 1
fi

# NVIDIA 3090 Configuration
# Total VRAM: 24GB
# Strategy: Run one model at a time or use model swapping
# For production: Use separate machines or time-based scheduling

# Port configuration
DEEPSEEK_PORT=8000
NANONETS_PORT=8001
GRANITE_PORT=8002

# Model paths (adjust if using local cache)
DEEPSEEK_MODEL="deepseek-ai/DeepSeek-OCR"
NANONETS_MODEL="nanonets/Nanonets-OCR2-3B"
GRANITE_MODEL="ibm-granite/granite-docling-258m"

# vLLM optimization for 3090 (24GB VRAM)
GPU_MEMORY_UTILIZATION=0.85  # Leave 15% for overhead
MAX_MODEL_LEN=8192  # Reduces memory usage
TENSOR_PARALLEL_SIZE=1  # Single GPU

echo -e "${YELLOW}=== Configuration ===${NC}"
echo "GPU: NVIDIA 3090 (24GB VRAM)"
echo "GPU Memory Utilization: ${GPU_MEMORY_UTILIZATION}"
echo "Max Model Length: ${MAX_MODEL_LEN}"
echo ""

# Function to start a vLLM server
start_vllm_server() {
    local MODEL=$1
    local PORT=$2
    local NAME=$3

    echo -e "${GREEN}Starting ${NAME} on port ${PORT}...${NC}"

    # Start vLLM server in background
    nohup vllm serve ${MODEL} \
        --host 0.0.0.0 \
        --port ${PORT} \
        --dtype bfloat16 \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --max-model-len ${MAX_MODEL_LEN} \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --disable-log-requests \
        --trust-remote-code \
        > logs/${NAME}.log 2>&1 &

    echo $! > logs/${NAME}.pid
    echo -e "${GREEN}${NAME} started (PID: $(cat logs/${NAME}.pid))${NC}"
    echo "  - URL: http://localhost:${PORT}"
    echo "  - Logs: logs/${NAME}.log"
    echo ""
}

# Create logs directory
mkdir -p logs

# Option 1: Sequential mode (recommended for 3090)
# Run one model at a time based on workload
echo -e "${YELLOW}=== Starting in SEQUENTIAL mode ===${NC}"
echo "For 3090, start ONE server at a time based on your workload:"
echo ""
echo "For standard documents (fast):"
echo "  ./scripts/start_vllm_servers.sh deepseek"
echo ""
echo "For handwriting/signatures/VQA:"
echo "  ./scripts/start_vllm_servers.sh nanonets"
echo ""
echo "For semantic enrichment:"
echo "  ./scripts/start_vllm_servers.sh granite"
echo ""

# Parse command line argument
if [ "$1" == "deepseek" ]; then
    start_vllm_server ${DEEPSEEK_MODEL} ${DEEPSEEK_PORT} "deepseek"
elif [ "$1" == "nanonets" ]; then
    start_vllm_server ${NANONETS_MODEL} ${NANONETS_PORT} "nanonets"
elif [ "$1" == "granite" ]; then
    start_vllm_server ${GRANITE_MODEL} ${GRANITE_PORT} "granite"
elif [ "$1" == "all" ]; then
    echo -e "${YELLOW}WARNING: Starting all models simultaneously${NC}"
    echo "This may exceed 24GB VRAM. Consider sequential mode instead."
    echo ""
    read -p "Continue? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Start Granite first (smallest, 258M)
        start_vllm_server ${GRANITE_MODEL} ${GRANITE_PORT} "granite"
        sleep 30

        # Then DeepSeek (3B)
        start_vllm_server ${DEEPSEEK_MODEL} ${DEEPSEEK_PORT} "deepseek"
        sleep 30

        # Then Nanonets (4B) - may cause OOM
        start_vllm_server ${NANONETS_MODEL} ${NANONETS_PORT} "nanonets"
    else
        echo "Cancelled."
        exit 0
    fi
else
    echo -e "${YELLOW}Usage: $0 [deepseek|nanonets|granite|all]${NC}"
    echo ""
    echo "Examples:"
    echo "  $0 deepseek    # Start DeepSeek-OCR only (recommended for standard docs)"
    echo "  $0 nanonets    # Start Nanonets-OCR2-3B only (for handwriting/signatures)"
    echo "  $0 granite     # Start Granite-Docling only (for enrichment)"
    echo "  $0 all         # Start all models (may exceed 24GB VRAM)"
    exit 1
fi

echo ""
echo -e "${GREEN}=== Servers Started ===${NC}"
echo ""
echo "Check status:"
echo "  ps aux | grep vllm"
echo ""
echo "Stop servers:"
echo "  ./scripts/stop_vllm_servers.sh"
echo ""
echo "Test endpoints:"
echo "  curl http://localhost:${DEEPSEEK_PORT}/v1/models"
echo "  curl http://localhost:${NANONETS_PORT}/v1/models"
echo "  curl http://localhost:${GRANITE_PORT}/v1/models"
