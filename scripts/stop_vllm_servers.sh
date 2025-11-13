#!/bin/bash
# Stop all vLLM servers

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping vLLM servers...${NC}"
echo ""

# Check for PID files
if [ -d "logs" ]; then
    for pidfile in logs/*.pid; do
        if [ -f "$pidfile" ]; then
            PID=$(cat "$pidfile")
            NAME=$(basename "$pidfile" .pid)

            if ps -p $PID > /dev/null 2>&1; then
                echo -e "${YELLOW}Stopping ${NAME} (PID: ${PID})...${NC}"
                kill $PID
                sleep 2

                # Force kill if still running
                if ps -p $PID > /dev/null 2>&1; then
                    echo -e "${YELLOW}Force stopping ${NAME}...${NC}"
                    kill -9 $PID
                fi

                echo -e "${GREEN}${NAME} stopped${NC}"
            else
                echo -e "${YELLOW}${NAME} not running (stale PID file)${NC}"
            fi

            rm "$pidfile"
        fi
    done
else
    echo "No PID files found in logs/ directory"
fi

# Also kill any remaining vllm processes
echo ""
echo -e "${YELLOW}Checking for remaining vLLM processes...${NC}"
pkill -f "vllm serve" || echo "No additional vLLM processes found"

echo ""
echo -e "${GREEN}All vLLM servers stopped${NC}"
