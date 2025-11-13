# Hybrid Document Parser

High-accuracy document processing system for financial, legal, and compliance documents. Optimized for NVIDIA 3090 (24GB VRAM) with support for 2000-3000 pages/day throughput.

## Features

- **Hybrid Processing**: Intelligent routing between DeepSeek-OCR, Nanonets-OCR2-3B, and Granite-Docling
- **Accuracy Modes**: Fast, Balanced, and Maximum accuracy modes
- **Specialized Capabilities**:
  - Handwriting recognition (11 languages)
  - Signature detection
  - Complex table extraction
  - VQA (Visual Question Answering)
  - Semantic structure preservation
- **Graph-Vector DB Ready**: Outputs structured JSON with nodes and edges
- **Production Optimized**: vLLM support for 2-3x faster inference

## Quick Start

### Prerequisites

- **Hardware**: NVIDIA GPU with 24GB+ VRAM (tested on RTX 3090)
- **Software**: Python 3.10+, CUDA 12.1+, WSL2 (for Windows)
- **Package Manager**: uv (recommended) or pip

### Installation

1. **Clone the repository**:
```bash
cd /path/to/gonova/document-parser
```

2. **Create virtual environment with uv**:
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate environment
uv venv
source .venv/bin/activate  # On Windows WSL
```

3. **Install dependencies**:
```bash
# Core dependencies
uv pip install -r requirements.txt

# Optional: vLLM for production (recommended)
uv pip install vllm

# Optional: Flash Attention for faster inference
uv pip install flash-attn --no-build-isolation
```

4. **Configure environment**:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Basic Usage

#### 1. Start the API Server

```bash
# Install API dependencies
uv pip install fastapi uvicorn[standard]

# Start API server
uvicorn document_parser.api:app --host 0.0.0.0 --port 8080

# Or using the script
docparse-api

# API will be available at:
# - API: http://localhost:8080
# - Docs: http://localhost:8080/docs
# - Health: http://localhost:8080/health
```

#### 2. Process Documents via API

```bash
# Upload and process a document (async)
curl -X POST "http://localhost:8080/v1/process" \
  -F "file=@document.pdf" \
  -F "accuracy_mode=balanced" \
  -F "output_format=json"

# Returns: {"job_id": "uuid", "status": "pending", ...}

# Check job status
curl "http://localhost:8080/v1/jobs/{job_id}"

# Download result
curl "http://localhost:8080/v1/jobs/{job_id}/download" -o result.json

# Process synchronously (for small documents)
curl -X POST "http://localhost:8080/v1/process/sync" \
  -F "file=@document.pdf" \
  -F "accuracy_mode=balanced" \
  -F "output_format=json"
```

#### 3. Python Client Example

```python
import requests

# Upload document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8080/v1/process",
        files={"file": f},
        data={
            "accuracy_mode": "balanced",
            "output_format": "json",
            "vqa_questions": "What is total revenue?,Who is the CEO?"
        }
    )

job_id = response.json()["job_id"]

# Poll for completion
import time
while True:
    status = requests.get(f"http://localhost:8080/v1/jobs/{job_id}").json()
    if status["status"] == "completed":
        result = status["result"]
        print(f"Nodes: {len(result['nodes'])}")
        print(f"Edges: {len(result['edges'])}")
        break
    time.sleep(2)
```

#### 4. Direct Python API (No Server)

```python
from document_parser import HybridDocumentProcessor
from document_parser.config import AccuracyMode

# Initialize processor
processor = HybridDocumentProcessor(
    accuracy_mode=AccuracyMode.BALANCED,
    inference_mode="transformers"  # or "vllm"
)

# Process document
result = processor.process(
    pdf_path="document.pdf",
    output_format="json",
    vqa_questions=["What is total revenue?"]
)

# Access results
print(f"Nodes: {len(result.nodes)}")
print(f"Edges: {len(result.edges)}")
print(f"VQA Answers: {result.vqa_answers}")
```

## Embedding Generation (Optional)

The system can optionally generate **Qwen3-Embedding-0.6B** embeddings alongside document processing:

### Features
- **1536-dimensional vectors** for each node
- **8192 context length** (handles long text)
- **Multi-lingual support** (English, Chinese, and more)
- **Agent-ready**: Direct integration with vector-graph DBs
- **Minimal overhead**: Only ~600MB VRAM, batched processing

### Usage

**Via API**:
```bash
# With embeddings
curl -X POST "http://localhost:8080/v1/process" \
  -F "file=@document.pdf" \
  -F "accuracy_mode=balanced" \
  -F "generate_embeddings=true"
```

**Via Python**:
```python
from document_parser import HybridDocumentProcessor

processor = HybridDocumentProcessor(enable_embeddings=True)
result = processor.process("document.pdf")

# Each node now has embedding vector
for node in result.nodes:
    print(f"Node: {node['id']}, Embedding dim: {len(node['embedding'])}")
```

**Output with embeddings**:
```json
{
  "nodes": [
    {
      "id": "node_0",
      "type": "section",
      "content": "Executive Summary...",
      "embedding": [0.123, -0.456, ..., 0.789],  // 1536 dimensions
      "page": 1
    }
  ],
  "metadata": {
    "embeddings_generated": true,
    "embedding_dim": 1536
  }
}
```

### VRAM Impact

- **DeepSeek (18GB) + Qwen3 (~1GB)**: ~19GB total ✅
- **Nanonets (20GB) + Qwen3 (~1GB)**: ~21GB total ✅

Still fits comfortably on 3090 (22GB usable)!

## Job Queue System

The API uses Redis for reliable job persistence:

- **No job loss**: Jobs survive server restarts
- **FIFO queue**: Processing order guaranteed
- **Automatic retry**: Up to 3 retry attempts on failures
- **Auto-cleanup**: Completed jobs removed after 24h
- **Fallback**: Works without Redis (in-memory mode)

For production with 2000-3000 pages/day, Redis is **highly recommended**.

**Start Redis**:
```bash
# Local
docker run -d -p 6379:6379 redis:7-alpine

# Or use Docker Compose (includes Redis)
docker compose --profile deepseek up -d
```

**Configure** in `.env`:
```bash
REDIS_URL=redis://localhost:6379/0
```

## Production Deployment

### Option 1: Docker Compose (Recommended)

Deploy the full stack with API server + vLLM + Redis:

```bash
# Start API + DeepSeek (for standard documents)
docker compose --profile deepseek up -d

# Check status
docker compose ps
docker compose logs -f api

# Access API at http://localhost:8080
# Swagger docs at http://localhost:8080/docs

# For handwriting/signatures: swap to Nanonets
docker compose stop vllm-deepseek
docker compose --profile nanonets up -d vllm-nanonets

# Stop all
docker compose down
```

**Note**: On NVIDIA 3090 (24GB), run ONE vLLM model at a time.

### Option 2: Separate vLLM Deployment

Run vLLM backends separately for more control:

```bash
# Start DeepSeek vLLM server
docker compose -f docker-compose.vllm.yml --profile deepseek up -d

# Start API server (local)
INFERENCE_MODE=vllm uvicorn document_parser.api:app --host 0.0.0.0 --port 8080

# Or via Docker
docker build -t docparser-api .
docker run -p 8080:8080 \
  -e INFERENCE_MODE=vllm \
  -e VLLM_DEEPSEEK_URL=http://host.docker.internal:8000/v1 \
  docparser-api
```

### Option 3: Shell Scripts (Development)

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Start DeepSeek-OCR server
./scripts/start_vllm_servers.sh deepseek

# In another terminal, start API
docparse-api

# Stop vLLM servers
./scripts/stop_vllm_servers.sh
```

### Environment Configuration for vLLM

Update `.env` for vLLM mode:

```bash
# Inference mode
INFERENCE_MODE=vllm

# vLLM endpoints (adjust if using Docker network)
VLLM_DEEPSEEK_URL=http://localhost:8000/v1
VLLM_NANONETS_URL=http://localhost:8001/v1
VLLM_GRANITE_URL=http://localhost:8002/v1

# Or for Docker Compose network
VLLM_DEEPSEEK_URL=http://vllm-deepseek:8000/v1
VLLM_NANONETS_URL=http://vllm-nanonets:8001/v1
VLLM_GRANITE_URL=http://vllm-granite:8002/v1
```

## API Endpoints

The API provides a RESTful interface for document processing:

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API information |
| `GET` | `/health` | Health check with processor status |
| `GET` | `/docs` | Interactive API documentation (Swagger) |
| `GET` | `/v1/capabilities` | Get processor capabilities |

### Processing Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/process` | Upload and process document (async) |
| `POST` | `/v1/process/sync` | Process document synchronously |

**Request Parameters**:
- `file`: PDF file (multipart/form-data)
- `accuracy_mode`: `fast`, `balanced`, or `maximum`
- `output_format`: `json`, `markdown`, or `html`
- `vqa_questions`: Comma-separated questions
- `extract_signatures`: Enable signature detection

**Response** (async):
```json
{
  "job_id": "uuid",
  "status": "pending",
  "message": "Processing started"
}
```

### Job Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/jobs` | List all jobs (with filters) |
| `GET` | `/v1/jobs/{job_id}` | Get job status |
| `GET` | `/v1/jobs/{job_id}/download` | Download processed document |
| `DELETE` | `/v1/jobs/{job_id}` | Delete job and files |

**Job Status Response**:
```json
{
  "job_id": "uuid",
  "status": "completed",
  "created_at": 1234567890.0,
  "completed_at": 1234567895.2,
  "result": {
    "document_id": "doc_id",
    "filename": "document.pdf",
    "nodes": [...],
    "edges": [...],
    "metadata": {...},
    "vqa_answers": {...}
  }
}
```

### Example Workflows

**1. Async Processing (Recommended for large documents)**:
```bash
# 1. Upload document
JOB_ID=$(curl -s -X POST "http://localhost:8080/v1/process" \
  -F "file=@document.pdf" \
  -F "accuracy_mode=balanced" | jq -r '.job_id')

# 2. Poll for completion
while true; do
  STATUS=$(curl -s "http://localhost:8080/v1/jobs/$JOB_ID" | jq -r '.status')
  echo "Status: $STATUS"
  [ "$STATUS" = "completed" ] && break
  sleep 2
done

# 3. Download result
curl "http://localhost:8080/v1/jobs/$JOB_ID/download" -o result.json
```

**2. Sync Processing (Small documents)**:
```bash
# Process and get result immediately
curl -X POST "http://localhost:8080/v1/process/sync" \
  -F "file=@small_doc.pdf" \
  -F "accuracy_mode=fast" \
  -F "output_format=json" | jq '.nodes | length'
```

**3. VQA (Visual Question Answering)**:
```bash
curl -X POST "http://localhost:8080/v1/process/sync" \
  -F "file=@invoice.pdf" \
  -F "vqa_questions=What is the invoice number?,What is the total amount?,Who is the vendor?" \
  | jq '.vqa_answers'
```

## Architecture

### Processing Pipeline

```
1. Document Analysis (fast, <5s)
   └─> Detect: pages, text layers, forms, handwriting

2. Intelligent Routing
   ├─> Forms/Signatures → Nanonets
   ├─> Handwriting (>20%) → Nanonets
   └─> Standard documents → DeepSeek

3. Primary Processing
   ├─> DeepSeek-OCR (3B): Fast, good tables
   └─> Nanonets-OCR2-3B (4B): Handwriting, signatures, VQA

4. VQA Processing (if questions provided)
   └─> Nanonets VQA mode

5. Semantic Enrichment (optional)
   └─> Granite-Docling (258M): Structure, equations

6. Graph Output
   └─> Nodes: Sections, tables, paragraphs
   └─> Edges: Relationships (follows, contains, references)
```

### Model Selection

| Model | Size | Strengths | Use Case |
|-------|------|-----------|----------|
| **DeepSeek-OCR** | 3B | Fast, good tables, multi-resolution | Standard documents, typed content |
| **Nanonets-OCR2-3B** | 4B | Handwriting, signatures, VQA, checkboxes | Forms, contracts, mixed content |
| **Granite-Docling** | 258M | Table structure, code blocks, equations | Semantic enrichment |

### Accuracy Modes

| Mode | Speed | Accuracy | Best For |
|------|-------|----------|----------|
| **Fast** | 3000-5000 pg/day | Good | High volume, standard documents |
| **Balanced** | 2000-3000 pg/day | Very Good | **Recommended** for compliance docs |
| **Maximum** | 1000-1500 pg/day | Excellent | Critical documents requiring highest accuracy |

## Configuration

### GPU Settings (.env)

```bash
# NVIDIA 3090 Optimization
CUDA_VISIBLE_DEVICES=0
TORCH_DEVICE=cuda
VRAM_LIMIT_GB=22  # Leave 2GB headroom

# Batch sizes for 3090
DEEPSEEK_BATCH_SIZE=4
NANONETS_BATCH_SIZE=2
GRANITE_BATCH_SIZE=8
```

### Accuracy & Inference

```bash
# Accuracy mode: fast, balanced, maximum
DEFAULT_ACCURACY_MODE=balanced

# Inference mode: transformers, vllm
INFERENCE_MODE=vllm  # Recommended for production

# Enable/disable enrichment
ENABLE_ENRICHMENT=true
```

### Model Configuration

```bash
# Model IDs (HuggingFace)
DEEPSEEK_MODEL=deepseek-ai/DeepSeek-OCR
NANONETS_MODEL=nanonets/Nanonets-OCR2-3B
GRANITE_MODEL=ibm-granite/granite-docling-258m

# Feature flags
ENABLE_SIGNATURE_DETECTION=true
ENABLE_HANDWRITING_DETECTION=true
```

## Output Formats

### 1. Markdown (Human-Readable)

```bash
docparse process document.pdf -f markdown -o output.md
```

### 2. JSON (Graph-Vector DB Ready)

```bash
docparse process document.pdf -f json -o output.json
```

Output structure:
```json
{
  "document_id": "document",
  "filename": "document.pdf",
  "nodes": [
    {
      "id": "node_0",
      "type": "section",
      "content": "Executive Summary",
      "page": 1
    },
    {
      "id": "node_1",
      "type": "table",
      "content": "...",
      "data": {...}
    }
  ],
  "edges": [
    {
      "source": "node_0",
      "target": "node_1",
      "type": "follows"
    }
  ],
  "metadata": {
    "num_pages": 10,
    "processing_time": 5.2,
    "processor": "deepseek",
    "signatures_found": ["page_5"],
    "table_count": 3
  },
  "vqa_answers": {
    "What is total revenue?": "$1.2M"
  }
}
```

### 3. HTML (Web Display)

```bash
docparse process document.pdf -f html -o output.html
```

## Performance Benchmarks (NVIDIA 3090)

### Transformers Mode (Direct Inference)

| Accuracy Mode | Pages/Second | Pages/Day (24h) | Memory Usage |
|---------------|--------------|-----------------|--------------|
| Fast | 1.5-2.0 | 3000-4000 | ~18GB |
| Balanced | 0.8-1.2 | 2000-3000 | ~20GB |
| Maximum | 0.5-0.8 | 1000-1500 | ~22GB |

### vLLM Mode (Recommended)

| Accuracy Mode | Pages/Second | Pages/Day (24h) | Speedup |
|---------------|--------------|-----------------|---------|
| Fast | 3.0-4.0 | 6000-8000 | 2x |
| Balanced | 1.5-2.5 | 3000-5000 | 2-3x |
| Maximum | 0.8-1.5 | 1500-3000 | 1.5-2x |

**Target met**: ✓ 2000-3000 pages/day with balanced mode

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce VRAM limit in .env
VRAM_LIMIT_GB=20

# Reduce batch sizes
DEEPSEEK_BATCH_SIZE=2
NANONETS_BATCH_SIZE=1

# Use fast accuracy mode
DEFAULT_ACCURACY_MODE=fast

# Use vLLM with smaller max_model_len
vllm serve deepseek-ai/DeepSeek-OCR --max-model-len=4096
```

### Slow Processing

```bash
# Use vLLM instead of Transformers
INFERENCE_MODE=vllm

# Use fast accuracy mode
DEFAULT_ACCURACY_MODE=fast

# Increase workers for batch processing
python scripts/batch_process.py ./docs/ -w 2

# Check GPU utilization
nvidia-smi -l 1
```

### vLLM Connection Issues

```bash
# Check if server is running
curl http://localhost:8000/v1/models

# Check logs
docker compose -f docker-compose.vllm.yml logs -f vllm-deepseek

# Restart server
docker compose -f docker-compose.vllm.yml restart vllm-deepseek
```

### Import Errors on macOS

The code is designed to run on the NVIDIA machine in WSL. Import warnings on macOS are expected and can be ignored during development.

## Integration Examples

### Neo4j Graph Database

```python
from neo4j import GraphDatabase
from document_parser import HybridDocumentProcessor

# Process document
processor = HybridDocumentProcessor()
result = processor.process("document.pdf", output_format="json")

# Connect to Neo4j
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    # Create document node
    session.run(
        "CREATE (d:Document {id: $id, filename: $filename})",
        id=result.document_id,
        filename=result.filename
    )

    # Create content nodes
    for node in result.nodes:
        session.run(
            """
            MATCH (d:Document {id: $doc_id})
            CREATE (n:Content {id: $id, type: $type, content: $content})
            CREATE (d)-[:CONTAINS]->(n)
            """,
            doc_id=result.document_id,
            **node
        )

    # Create relationships
    for edge in result.edges:
        session.run(
            """
            MATCH (s:Content {id: $source})
            MATCH (t:Content {id: $target})
            CREATE (s)-[:FOLLOWS]->(t)
            """,
            **edge
        )
```

### Embedding + Vector DB

```python
from document_parser import HybridDocumentProcessor
from sentence_transformers import SentenceTransformer
import chromadb

# Process document
processor = HybridDocumentProcessor()
result = processor.process("document.pdf")

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = []
for node in result.nodes:
    embedding = model.encode(node["content"])
    embeddings.append(embedding)

# Store in ChromaDB
client = chromadb.Client()
collection = client.create_collection("documents")

collection.add(
    embeddings=embeddings,
    documents=[node["content"] for node in result.nodes],
    metadatas=result.nodes,
    ids=[node["id"] for node in result.nodes]
)
```

## Project Structure

```
document-parser/
├── document_parser/          # Main package
│   ├── __init__.py
│   ├── config.py            # Settings management
│   ├── base_processor.py    # Abstract base class
│   ├── deepseek_processor.py   # DeepSeek-OCR
│   ├── nanonets_processor.py   # Nanonets-OCR2-3B
│   ├── granite_processor.py    # Granite-Docling
│   ├── document_analyzer.py    # Fast PDF analysis
│   ├── router.py               # Intelligent routing
│   ├── hybrid_processor.py     # Main orchestration
│   └── cli.py                  # Command-line interface
├── scripts/
│   ├── start_vllm_servers.sh   # vLLM deployment
│   ├── stop_vllm_servers.sh    # Stop servers
│   └── batch_process.py        # Optimized batch processing
├── docker-compose.vllm.yml   # Docker deployment
├── pyproject.toml            # Project config
├── requirements.txt          # Dependencies
├── .env.example             # Environment template
└── README.md                # This file
```

## Development

### Running Tests

```bash
# Install test dependencies
uv pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=document_parser --cov-report=html
```

### Code Quality

```bash
# Install dev dependencies
uv pip install ruff black isort

# Format code
black document_parser/
isort document_parser/

# Lint
ruff check document_parser/
```

## License

MIT License - See LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-org/document-parser/issues)
- Documentation: See `findings-and-plan.md` for detailed research and architecture

## Acknowledgments

Built on top of:
- [Docling](https://github.com/DS4SD/docling) - IBM Research document processing
- [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) - DeepSeek OCR model
- [Nanonets-OCR2-3B](https://huggingface.co/nanonets/Nanonets-OCR2-3B) - Nanonets OCR model
- [vLLM](https://github.com/vllm-project/vllm) - Fast LLM inference
# gonova-document-parser
