#!/bin/bash
# Quick start script for Document Parser API
# This script sets up and runs the API server with vLLM backend

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Document Parser Quick Start${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if running on NVIDIA machine
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}WARNING: nvidia-smi not found. Make sure you're on the NVIDIA machine.${NC}"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python version: ${PYTHON_VERSION}${NC}"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv .venv
fi

# Activate venv
echo -e "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Install dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
if command -v uv &> /dev/null; then
    echo "Using uv (faster)"
    uv pip install -r requirements.txt
else
    echo "Using pip"
    pip install -r requirements.txt
fi

# Create .env if not exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}.env not found. Creating from template...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}Please edit .env with your configuration${NC}"
fi

# Create required directories
mkdir -p uploads outputs logs

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Choose Deployment Mode${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "1) Docker Compose (Recommended - Full stack with vLLM)"
echo "2) Local API + Docker vLLM (API local, vLLM in Docker)"
echo "3) Local API only (Transformers mode, no Docker)"
echo "4) Exit"
echo ""
read -p "Select option (1-4): " choice

case $choice in
    1)
        echo ""
        echo -e "${GREEN}Starting with Docker Compose...${NC}"
        echo ""
        echo "Which vLLM model to use?"
        echo "1) DeepSeek-OCR (recommended for standard documents)"
        echo "2) Nanonets-OCR2-3B (for handwriting/signatures)"
        echo ""
        read -p "Select model (1-2): " model_choice

        if [ "$model_choice" = "1" ]; then
            PROFILE="deepseek"
        else
            PROFILE="nanonets"
        fi

        echo -e "${GREEN}Starting Docker Compose with ${PROFILE}...${NC}"
        docker compose --profile $PROFILE up -d

        echo ""
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}  API Server Started!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo ""
        echo "API: http://localhost:8080"
        echo "Docs: http://localhost:8080/docs"
        echo "Health: http://localhost:8080/health"
        echo ""
        echo "Check logs:"
        echo "  docker compose logs -f api"
        echo ""
        echo "Stop server:"
        echo "  docker compose down"
        ;;

    2)
        echo ""
        echo -e "${GREEN}Starting vLLM in Docker...${NC}"
        echo ""
        echo "Which model?"
        echo "1) DeepSeek-OCR"
        echo "2) Nanonets-OCR2-3B"
        echo ""
        read -p "Select model (1-2): " model_choice

        if [ "$model_choice" = "1" ]; then
            PROFILE="deepseek"
        else
            PROFILE="nanonets"
        fi

        docker compose -f docker-compose.vllm.yml --profile $PROFILE up -d

        echo ""
        echo -e "${GREEN}vLLM server started. Starting local API...${NC}"

        # Update .env for vLLM mode
        if grep -q "INFERENCE_MODE=transformers" .env 2>/dev/null; then
            sed -i.bak 's/INFERENCE_MODE=transformers/INFERENCE_MODE=vllm/' .env
        fi

        echo ""
        echo -e "${GREEN}Starting API server...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""

        uvicorn document_parser.api:app --host 0.0.0.0 --port 8080
        ;;

    3)
        echo ""
        echo -e "${GREEN}Starting API in Transformers mode (no Docker)...${NC}"
        echo -e "${YELLOW}This will use more VRAM and be slower than vLLM${NC}"
        echo ""

        # Update .env for transformers mode
        if grep -q "INFERENCE_MODE=vllm" .env 2>/dev/null; then
            sed -i.bak 's/INFERENCE_MODE=vllm/INFERENCE_MODE=transformers/' .env
        fi

        echo -e "${GREEN}Starting API server...${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""

        uvicorn document_parser.api:app --host 0.0.0.0 --port 8080
        ;;

    4)
        echo "Exiting..."
        exit 0
        ;;

    *)
        echo -e "${YELLOW}Invalid option${NC}"
        exit 1
        ;;
esac
