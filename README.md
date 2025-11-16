# Document Parser API

High-accuracy document processing API that connects to external vLLM servers for OCR and markdown conversion.

## What This Does

Intelligently routes PDF documents to specialized OCR models and returns beautiful markdown:

- **DeepSeek-OCR** (port 4444): Fast processing for standard documents, tables, charts
- **Nanonets-OCR2-3B** (port 4445): Handwriting, signatures, forms, VQA
- **Granite-Docling** (port 4446): Semantic enrichment (optional)

## Prerequisites

**This application does NOT deploy vLLM servers - they must be running externally.**

### Required Services

1. **vLLM Server(s)** - At least one running on your host machine:
   ```bash
   # DeepSeek-OCR on port 4444 (recommended)
   vllm serve deepseek-ai/DeepSeek-OCR --port 4444 ...
   ```
   See [PREREQUISITES.md](PREREQUISITES.md) for full vLLM setup instructions.

2. **Redis** - For job queue:
   ```bash
   redis-server
   # Or: docker run -d -p 6379:6379 redis:7-alpine
   ```

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[api,db]"

# 2. Copy and configure environment
cp .env.example .env
# Edit .env to point to your vLLM servers (default: localhost:4444/4445/4446)

# 3. Verify external services are running
curl http://localhost:4444/v1/models  # vLLM
redis-cli ping                         # Redis

# 4. Start the API
./start_api.sh
# Or: uvicorn document_parser.api:app --host 0.0.0.0 --port 1233
```

## Usage

### Interactive API Docs
Open in browser: http://localhost:1233/docs

### Upload a PDF
```bash
curl -X POST "http://localhost:1233/v1/process/sync" \
  -F "file=@document.pdf" \
  -F "accuracy_mode=balanced" \
  -F "output_format=markdown" \
  | jq -r '.metadata.output' > output.md
```

### With VQA (Visual Question Answering)
```bash
curl -X POST "http://localhost:1233/v1/process/sync" \
  -F "file=@invoice.pdf" \
  -F "vqa_questions=What is the total amount?,Who is the vendor?" \
  | jq '.vqa_answers'
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/docs` | GET | Interactive API documentation |
| `/v1/process` | POST | Upload PDF (async processing) |
| `/v1/process/sync` | POST | Upload PDF (wait for result) |
| `/v1/jobs/{job_id}` | GET | Get job status |
| `/v1/jobs/{job_id}/download` | GET | Download processed result |

## Architecture

```
PDF Upload → Fast Analysis → Intelligent Routing → vLLM Processing → Beautiful Markdown
             (pypdf)         (DeepSeek/Nanonets)   (External)        (Structured output)
```

### Processing Pipeline

1. **Fast Analysis** (< 1s): Analyzes PDF metadata without OCR
2. **Intelligent Routing**: Selects best processor based on document type
3. **Parallel Processing**: Processes 32 pages concurrently (DeepSeek) or 16 pages (Nanonets)
4. **Graph Output**: Returns nodes/edges for RAG/vector DB ingestion

## Performance

With external vLLM servers (optimized):

| Document Size | Processing Time | Throughput |
|---------------|----------------|------------|
| 10 pages | ~4s | **6-8x faster** than sequential |
| 50 pages | ~12s | ~4 pages/sec |
| 100 pages | ~20s | ~5 pages/sec |

*Using DeepSeek with 32-page parallel batching*

## Configuration

Key settings in `.env`:

```bash
# External vLLM servers (REQUIRED)
VLLM_DEEPSEEK_URL=http://localhost:4444/v1/chat/completions
VLLM_NANONETS_URL=http://localhost:4445/v1/chat/completions
VLLM_GRANITE_URL=http://localhost:4446/v1/chat/completions

# Processing settings
INFERENCE_MODE=vllm                    # Always use vllm mode
DEFAULT_ACCURACY_MODE=balanced         # fast, balanced, or maximum
ENABLE_ENRICHMENT=false               # Granite enrichment (slower)

# API server
API_HOST=0.0.0.0
API_PORT=1233
```

## Output Format

Returns structured JSON with:

```json
{
  "document_id": "invoice",
  "filename": "invoice.pdf",
  "nodes": [
    {"id": "node_0", "type": "section", "content": "...", "page": 1},
    {"id": "node_1", "type": "table", "content": "...", "page": 2}
  ],
  "edges": [
    {"source": "node_0", "target": "node_1", "type": "follows"}
  ],
  "metadata": {
    "num_pages": 5,
    "processing_time": 4.2,
    "processor": "deepseek",
    "output": "# Full Markdown Content\n\n..."
  },
  "vqa_answers": {"What is the total?": "$1,234.56"}
}
```

## Documentation

- [PREREQUISITES.md](PREREQUISITES.md) - External service setup
- [OPTIMIZATIONS.md](OPTIMIZATIONS.md) - Performance optimizations explained
- [API Docs](http://localhost:1233/docs) - Interactive API documentation

## What This App Does NOT Do

- ❌ Deploy vLLM servers (they run externally)
- ❌ Load AI models into GPU (models run on your vLLM servers)
- ❌ Require GPU access (API server is CPU-only)

## License

MIT License
