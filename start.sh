#!/bin/bash

# Document Parser - Production Startup Script
# Checks dependencies and starts the API with background workers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================"
echo "  Document Parser - Starting Production Server"
echo "======================================================"
echo ""

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo -e "${GREEN}✓${NC} Loaded .env configuration"
else
    echo -e "${YELLOW}⚠${NC} No .env file found, using defaults"
fi

# Check Redis
echo ""
echo "Checking dependencies..."
if redis-cli ping > /dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} Redis is running"
else
    echo -e "${RED}✗${NC} Redis is NOT running"
    echo "  Please start Redis: sudo systemctl start redis"
    echo "  Or run: redis-server &"
    exit 1
fi

# Check vLLM servers
DEEPSEEK_URL=${VLLM_DEEPSEEK_URL:-"http://localhost:4444/v1"}
NANONETS_URL=${VLLM_NANONETS_URL:-"http://localhost:4445/v1"}

# Extract host and port from URLs
DEEPSEEK_PORT=$(echo $DEEPSEEK_URL | grep -oP ':\K[0-9]+' | head -1)
NANONETS_PORT=$(echo $NANONETS_URL | grep -oP ':\K[0-9]+' | head -1)

if nc -z localhost ${DEEPSEEK_PORT:-4444} 2>/dev/null; then
    echo -e "${GREEN}✓${NC} DeepSeek-OCR server is running (port ${DEEPSEEK_PORT:-4444})"
else
    echo -e "${YELLOW}⚠${NC} DeepSeek-OCR server not detected (port ${DEEPSEEK_PORT:-4444})"
    echo "  Make sure vLLM is running with DeepSeek model"
fi

if nc -z localhost ${NANONETS_PORT:-4445} 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Nanonets-OCR server is running (port ${NANONETS_PORT:-4445})"
else
    echo -e "${YELLOW}⚠${NC} Nanonets-OCR server not detected (port ${NANONETS_PORT:-4445})"
    echo "  Make sure vLLM is running with Nanonets model"
fi

# Check Python environment
if ! command -v python &> /dev/null; then
    echo -e "${RED}✗${NC} Python not found"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION"

# Check if dependencies are installed
if ! python -c "import redis" 2>/dev/null; then
    echo -e "${RED}✗${NC} Redis Python package not installed"
    echo "  Run: pip install redis rq"
    exit 1
fi

if ! python -c "import rq" 2>/dev/null; then
    echo -e "${RED}✗${NC} RQ package not installed"
    echo "  Run: pip install redis rq"
    exit 1
fi

echo -e "${GREEN}✓${NC} All dependencies installed"

# Get configuration
NUM_WORKERS=${NUM_WORKERS:-2}
API_PORT=${API_PORT:-1233}
API_HOST=${API_HOST:-0.0.0.0}

echo ""
echo "Configuration:"
echo "  API: http://${API_HOST}:${API_PORT}"
echo "  Workers: ${NUM_WORKERS}"
echo "  Redis: ${REDIS_URL:-redis://localhost:6379/0}"
echo ""

# Start the server
echo "======================================================"
echo "  Starting API Server with ${NUM_WORKERS} workers"
echo "======================================================"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Trap Ctrl+C to cleanup
trap 'echo ""; echo "Shutting down..."; exit 0' INT

# Start the API (workers start automatically)
python -m document_parser.api
