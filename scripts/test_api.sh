#!/bin/bash
# Test script for Document Parser API

set -e

API_URL="${1:-http://localhost:8080}"
TEST_PDF="${2:-test.pdf}"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "Testing Document Parser API at ${API_URL}"
echo ""

# Test 1: Health check
echo -e "${YELLOW}Test 1: Health Check${NC}"
HEALTH=$(curl -s "${API_URL}/health")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Health check passed${NC}"
    echo "$HEALTH" | jq '.'
else
    echo -e "${RED}✗ Health check failed${NC}"
    exit 1
fi
echo ""

# Test 2: Get capabilities
echo -e "${YELLOW}Test 2: Get Capabilities${NC}"
CAPABILITIES=$(curl -s "${API_URL}/v1/capabilities")
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Capabilities retrieved${NC}"
    echo "$CAPABILITIES" | jq '.'
else
    echo -e "${RED}✗ Failed to get capabilities${NC}"
    exit 1
fi
echo ""

# Test 3: Process document (if PDF provided)
if [ -f "$TEST_PDF" ]; then
    echo -e "${YELLOW}Test 3: Process Document (Async)${NC}"

    # Upload document
    RESPONSE=$(curl -s -X POST "${API_URL}/v1/process" \
        -F "file=@${TEST_PDF}" \
        -F "accuracy_mode=fast" \
        -F "output_format=json")

    JOB_ID=$(echo "$RESPONSE" | jq -r '.job_id')

    if [ "$JOB_ID" != "null" ] && [ -n "$JOB_ID" ]; then
        echo -e "${GREEN}✓ Document uploaded, job_id: ${JOB_ID}${NC}"

        # Poll for completion
        echo "Waiting for processing..."
        for i in {1..30}; do
            STATUS=$(curl -s "${API_URL}/v1/jobs/${JOB_ID}" | jq -r '.status')
            echo "  Status: $STATUS"

            if [ "$STATUS" = "completed" ]; then
                echo -e "${GREEN}✓ Processing completed${NC}"

                # Get result
                RESULT=$(curl -s "${API_URL}/v1/jobs/${JOB_ID}")
                echo "Result summary:"
                echo "$RESULT" | jq '{
                    document_id: .result.document_id,
                    filename: .result.filename,
                    node_count: (.result.nodes | length),
                    edge_count: (.result.edges | length),
                    processing_time: .result.metadata.processing_time
                }'
                break
            elif [ "$STATUS" = "failed" ]; then
                echo -e "${RED}✗ Processing failed${NC}"
                curl -s "${API_URL}/v1/jobs/${JOB_ID}" | jq '.error'
                exit 1
            fi

            sleep 2
        done
    else
        echo -e "${RED}✗ Failed to upload document${NC}"
        echo "$RESPONSE"
        exit 1
    fi
else
    echo -e "${YELLOW}Test 3: Skipped (no PDF provided)${NC}"
    echo "Usage: $0 <api_url> <test.pdf>"
fi
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All tests passed!${NC}"
echo -e "${GREEN}========================================${NC}"
