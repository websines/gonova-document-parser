#!/bin/bash
# Start the Document Parser API server
# Connects to external vLLM servers at localhost:4444, 4445, 4446

set -e

echo "🚀 Starting Document Parser API..."
echo ""
echo "📋 Prerequisites:"
echo "   ✓ vLLM servers running at:"
echo "     - DeepSeek-OCR: localhost:4444/v1"
echo "     - Nanonets: localhost:4445/v1"
echo "     - Granite: localhost:4446/v1"
echo "   ✓ Redis running at: localhost:6379"
echo ""
echo "🌐 API will be available at:"
echo "   - Main API: http://localhost:1233"
echo "   - Swagger Docs: http://localhost:1233/docs"
echo "   - Health: http://localhost:1233/health"
echo ""

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "⚠️  Warning: Redis is not responding at localhost:6379"
    echo "   The API will work but jobs won't persist across restarts"
    echo ""
fi

# Start the API (port from .env or override with: --port 1233)
uvicorn document_parser.api:app --host 0.0.0.0 --port 1233 --log-level info
