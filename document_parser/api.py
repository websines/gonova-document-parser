"""FastAPI server for document processing.

Production-ready API with:
- Async processing for high throughput
- Background task queue for long-running jobs
- Health checks and status endpoints
- OpenAPI/Swagger documentation
"""

import asyncio
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from document_parser.config import AccuracyMode, InferenceMode, settings
from document_parser.hybrid_processor import GraphDocument, HybridDocumentProcessor

# ============================================================================
# Models
# ============================================================================


class ProcessingStatus(str, Enum):
    """Processing job status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessDocumentRequest(BaseModel):
    """Request to process a document."""

    accuracy_mode: Optional[str] = Field(
        default="balanced",
        description="Accuracy mode: fast, balanced, or maximum",
    )
    output_format: Optional[str] = Field(
        default="json",
        description="Output format: markdown, json, or html",
    )
    vqa_questions: Optional[List[str]] = Field(
        default=None,
        description="Optional VQA questions to answer",
    )
    extract_signatures: Optional[bool] = Field(
        default=None,
        description="Enable signature detection",
    )
    enable_enrichment: Optional[bool] = Field(
        default=True,
        description="Enable Granite semantic enrichment",
    )


class ProcessDocumentResponse(BaseModel):
    """Response from document processing request."""

    job_id: str = Field(..., description="Unique job identifier")
    status: ProcessingStatus = Field(..., description="Current status")
    message: str = Field(..., description="Status message")


class JobStatusResponse(BaseModel):
    """Job status response."""

    job_id: str
    status: ProcessingStatus
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    progress: Optional[Dict] = None
    result: Optional[GraphDocument] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    processors: Dict[str, bool]
    config: Dict[str, str]


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Hybrid Document Parser API",
    description="High-accuracy document processing for financial, legal, and compliance documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
processor: Optional[HybridDocumentProcessor] = None
job_queue = None  # Will be initialized on startup
upload_dir = Path("./uploads")
output_dir = Path("./outputs")

# Create directories
upload_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)


# ============================================================================
# Startup/Shutdown
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Initialize processor and queue on startup."""
    global processor, job_queue
    logger.info("Starting document processing API...")

    # Initialize job queue
    from document_parser.queue import JobQueue

    redis_url = settings.redis_url if hasattr(settings, "redis_url") else "redis://localhost:6379/0"
    job_queue = JobQueue(redis_url=redis_url)

    logger.info(f"Job queue initialized (using Redis: {job_queue.use_redis})")

    # Initialize processor (embeddings off by default, enabled per-request)
    processor = HybridDocumentProcessor(
        accuracy_mode=settings.default_accuracy_mode,
        inference_mode=settings.inference_mode,
        enable_enrichment=True,
        enable_embeddings=False,  # Will be enabled per-request
    )

    logger.success("API ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global processor
    if processor:
        processor.cleanup()
    logger.info("API shutdown complete")


# ============================================================================
# Helper Functions
# ============================================================================


async def process_document_task(job_id: str, pdf_path: Path):
    """Background task to process document."""
    global job_queue

    job = job_queue.get_job(job_id)
    if not job:
        logger.error(f"Job {job_id} not found")
        return

    try:
        # Update status
        from document_parser.queue import JobStatus

        job.status = JobStatus.PROCESSING
        job.started_at = time.time()
        job_queue.update_job(job)

        logger.info(f"[{job_id}] Processing {pdf_path.name}...")

        # Process document (handle embeddings separately to avoid global state issues)
        # Temporarily enable embeddings if requested
        generate_embeddings = job.metadata.get("generate_embeddings", False) if hasattr(job, "metadata") and job.metadata else False

        original_embeddings_setting = processor.enable_embeddings
        processor.enable_embeddings = generate_embeddings

        try:
            result: GraphDocument = processor.process(
                pdf_path=pdf_path,
                output_format=job.output_format,
                accuracy_mode=AccuracyMode(job.accuracy_mode),
                vqa_questions=job.vqa_questions,
                extract_signatures=job.extract_signatures,
            )
        finally:
            # Restore original setting
            processor.enable_embeddings = original_embeddings_setting

        # Save output
        ext = {"markdown": "md", "json": "json", "html": "html"}[job.output_format]
        output_path = output_dir / f"{job_id}.{ext}"

        if job.output_format == "json":
            import json

            output_data = {
                "document_id": result.document_id,
                "filename": result.filename,
                "nodes": result.nodes,
                "edges": result.edges,
                "metadata": result.metadata,
                "vqa_answers": result.vqa_answers,
            }
            output_path.write_text(json.dumps(output_data, indent=2))
        else:
            content = result.metadata.get("output", "")
            output_path.write_text(content)

        # Update job status
        job.status = JobStatus.COMPLETED
        job.completed_at = time.time()
        job.result = {
            "document_id": result.document_id,
            "filename": result.filename,
            "nodes": result.nodes,
            "edges": result.edges,
            "metadata": result.metadata,
            "vqa_answers": result.vqa_answers,
            "output_path": str(output_path),
        }
        job_queue.update_job(job)

        logger.success(
            f"[{job_id}] Completed in {job.completed_at - job.started_at:.1f}s"
        )

    except Exception as e:
        logger.error(f"[{job_id}] Failed: {e}")
        job.status = JobStatus.FAILED
        job.completed_at = time.time()
        job.error = str(e)
        job_queue.update_job(job)


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint."""
    return {
        "service": "Hybrid Document Parser API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Info"])
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if processor else "initializing",
        version="1.0.0",
        processors=processor.get_status()["processors_loaded"] if processor else {},
        config={
            "accuracy_mode": settings.default_accuracy_mode.value,
            "inference_mode": settings.inference_mode.value,
            "device": settings.torch_device,
        },
    )


@app.post(
    "/v1/process",
    response_model=ProcessDocumentResponse,
    tags=["Processing"],
    summary="Process a PDF document",
    description="""
    Upload and process a PDF document with hybrid processing.

    Returns a job ID for tracking progress. Use /v1/jobs/{job_id} to check status.

    **Accuracy Modes:**
    - `fast`: 3000-5000 pages/day, good accuracy
    - `balanced`: 2000-3000 pages/day, very good accuracy (recommended)
    - `maximum`: 1000-1500 pages/day, excellent accuracy

    **Output Formats:**
    - `json`: Graph-ready structure with nodes and edges
    - `markdown`: Human-readable markdown
    - `html`: Web-ready HTML
    """,
)
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF file to process"),
    accuracy_mode: str = Form("balanced", description="Accuracy mode"),
    output_format: str = Form("json", description="Output format"),
    vqa_questions: Optional[str] = Form(
        None, description="Comma-separated VQA questions"
    ),
    extract_signatures: Optional[bool] = Form(None, description="Extract signatures"),
    generate_embeddings: bool = Form(False, description="Generate Qwen3 embeddings"),
):
    """Process a PDF document."""
    # Validate file
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    pdf_path = upload_dir / f"{job_id}.pdf"
    content = await file.read()
    pdf_path.write_bytes(content)

    # Parse VQA questions
    vqa_list = None
    if vqa_questions:
        vqa_list = [q.strip() for q in vqa_questions.split(",")]

    # Create job
    from document_parser.queue import Job, JobStatus

    job = Job(
        job_id=job_id,
        filename=file.filename,
        status=JobStatus.PENDING,
        created_at=time.time(),
        accuracy_mode=accuracy_mode,
        output_format=output_format,
        vqa_questions=vqa_list,
        extract_signatures=extract_signatures,
        metadata={"generate_embeddings": generate_embeddings},
    )

    # Add to queue
    job_queue.add_job(job)

    # Start background processing
    background_tasks.add_task(process_document_task, job_id, pdf_path)

    return ProcessDocumentResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Processing started for {file.filename}",
    )


@app.get(
    "/v1/jobs/{job_id}",
    response_model=JobStatusResponse,
    tags=["Jobs"],
    summary="Get job status",
    description="Check the status of a processing job",
)
async def get_job_status(job_id: str):
    """Get job status."""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    from document_parser.queue import JobStatus

    # Calculate progress
    progress = None
    if job.status == JobStatus.PROCESSING and job.started_at:
        elapsed = time.time() - job.started_at
        progress = {
            "elapsed_seconds": elapsed,
            "status": "Processing document...",
        }

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        progress=progress,
        result=job.result,
        error=job.error,
    )


@app.get(
    "/v1/jobs/{job_id}/download",
    tags=["Jobs"],
    summary="Download processed document",
    description="Download the processed document output",
)
async def download_result(job_id: str):
    """Download processed document."""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    from document_parser.queue import JobStatus

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    if not job.result or "output_path" not in job.result:
        raise HTTPException(status_code=404, detail="Output file not found")

    output_path = Path(job.result["output_path"])
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    return FileResponse(
        output_path,
        media_type="application/octet-stream",
        filename=f"{job.filename}.{output_path.suffix}",
    )


@app.get(
    "/v1/jobs",
    tags=["Jobs"],
    summary="List all jobs",
    description="Get list of all processing jobs",
)
async def list_jobs(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(100, description="Max results"),
):
    """List all jobs."""
    jobs_list = job_queue.list_jobs(status=status, limit=limit)

    # Convert to dict for JSON response
    jobs_data = [job.model_dump() for job in jobs_list]

    stats = job_queue.get_stats()

    return {
        "total": stats.get("total_jobs", 0),
        "filtered": len(jobs_list),
        "queue_length": stats.get("queue_length", 0),
        "stats": stats,
        "jobs": jobs_data,
    }


@app.delete(
    "/v1/jobs/{job_id}",
    tags=["Jobs"],
    summary="Delete a job",
    description="Delete a job and its associated files",
)
async def delete_job(job_id: str):
    """Delete a job."""
    job = job_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete files
    pdf_path = upload_dir / f"{job_id}.pdf"
    if pdf_path.exists():
        pdf_path.unlink()

    if job.result and "output_path" in job.result:
        output_path = Path(job.result["output_path"])
        if output_path.exists():
            output_path.unlink()

    # Delete job from queue
    job_queue.delete_job(job_id)

    return {"message": "Job deleted successfully"}


@app.post(
    "/v1/process/sync",
    tags=["Processing"],
    summary="Process document synchronously",
    description="""
    Process a document and wait for the result (synchronous).

    Use this for small documents or when you need immediate results.
    For large documents, use the async endpoint /v1/process instead.
    """,
)
async def process_document_sync(
    file: UploadFile = File(...),
    accuracy_mode: str = Form("balanced"),
    output_format: str = Form("json"),
    vqa_questions: Optional[str] = Form(None),
    extract_signatures: Optional[bool] = Form(None),
    generate_embeddings: bool = Form(False),
):
    """Process document synchronously."""
    # Validate file
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save file temporarily
    temp_id = str(uuid.uuid4())
    pdf_path = upload_dir / f"{temp_id}.pdf"
    content = await file.read()
    pdf_path.write_bytes(content)

    try:
        # Parse VQA questions
        vqa_list = None
        if vqa_questions:
            vqa_list = [q.strip() for q in vqa_questions.split(",")]

        # Process document (temporarily enable embeddings if requested)
        original_embeddings_setting = processor.enable_embeddings
        processor.enable_embeddings = generate_embeddings

        try:
            result: GraphDocument = processor.process(
                pdf_path=pdf_path,
                output_format=output_format,
                accuracy_mode=AccuracyMode(accuracy_mode),
                vqa_questions=vqa_list,
                extract_signatures=extract_signatures,
            )
        finally:
            processor.enable_embeddings = original_embeddings_setting

        # Return result
        return {
            "success": True,
            "document_id": result.document_id,
            "filename": result.filename,
            "nodes": result.nodes,
            "edges": result.edges,
            "metadata": result.metadata,
            "vqa_answers": result.vqa_answers,
        }

    except Exception as e:
        logger.error(f"Sync processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup temp file
        if pdf_path.exists():
            pdf_path.unlink()


@app.get(
    "/v1/capabilities",
    tags=["Info"],
    summary="Get processor capabilities",
    description="Get detailed information about processor capabilities",
)
async def get_capabilities():
    """Get processor capabilities."""
    if not processor:
        raise HTTPException(status_code=503, detail="Processor not initialized")

    return {
        "deepseek": processor.deepseek.get_capabilities(),
        "nanonets": processor.nanonets.get_capabilities(),
        "granite": processor.granite.get_capabilities(),
    }


# ============================================================================
# Main
# ============================================================================


def main():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "document_parser.api:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
