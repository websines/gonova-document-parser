"""Background worker for processing PDF jobs from Redis queue."""

import asyncio
import os
import sys
from pathlib import Path

from loguru import logger
from rq import Worker, Queue
import redis

from document_parser.config import AccuracyMode, settings
from document_parser.hybrid_processor import HybridDocumentProcessor
from document_parser.job_manager import JobManager


# Configure logger
logger.remove()
logger.add(sys.stderr, level=settings.log_level)


def process_pdf_job(
    job_id: str,
    pdf_path: str,
    accuracy_mode: str = "balanced",
    output_format: str = "markdown",
    generate_embeddings: bool = False,
):
    """
    Process a PDF file and store result in Redis.

    This function is executed by RQ workers.

    Args:
        job_id: Unique job identifier
        pdf_path: Path to uploaded PDF file
        accuracy_mode: Processing accuracy mode
        output_format: Output format (markdown, json, html)
        generate_embeddings: Whether to generate embeddings

    Returns:
        Job result dict
    """
    job_manager = JobManager()

    try:
        # Update status to processing
        job_manager.update_status(job_id, "processing")

        logger.info(f"[Job {job_id}] Starting processing: {pdf_path}")

        # Initialize processor
        processor = HybridDocumentProcessor(
            accuracy_mode=AccuracyMode(accuracy_mode),
            enable_embeddings=generate_embeddings,
        )

        # Process PDF (async function, so we need to run it in event loop)
        result = asyncio.run(
            processor.process(
                pdf_path=pdf_path,
                output_format=output_format,
            )
        )

        # Extract markdown output
        markdown_output = result.metadata.get("output", "")

        # Store result in Redis
        job_manager.store_result(job_id, markdown_output)

        # Update status to completed with metadata
        job_manager.update_status(
            job_id,
            "completed",
            progress={
                "total_pages": result.metadata.get("num_pages", 0),
                "processing_time": result.metadata.get("processing_time", 0),
                "deepseek_pages": result.metadata.get("deepseek_pages", 0),
                "nanonets_pages": result.metadata.get("nanonets_pages", 0),
            },
        )

        logger.success(
            f"[Job {job_id}] Completed: {result.metadata.get('num_pages', 0)} pages "
            f"in {result.metadata.get('processing_time', 0):.1f}s"
        )

        # Cleanup PDF file
        try:
            os.unlink(pdf_path)
            logger.debug(f"[Job {job_id}] Cleaned up temp file: {pdf_path}")
        except Exception as e:
            logger.warning(f"[Job {job_id}] Failed to cleanup {pdf_path}: {e}")

        return {
            "job_id": job_id,
            "status": "completed",
            "pages": result.metadata.get("num_pages", 0),
            "processing_time": result.metadata.get("processing_time", 0),
        }

    except Exception as e:
        logger.error(f"[Job {job_id}] Processing failed: {e}", exc_info=True)

        # Update status to failed
        job_manager.update_status(job_id, "failed", error=str(e))

        # Cleanup PDF file on error
        try:
            if os.path.exists(pdf_path):
                os.unlink(pdf_path)
        except Exception:
            pass

        # Re-raise for RQ to mark job as failed
        raise


def start_worker(queue_names: list[str] = None, burst: bool = False):
    """
    Start RQ worker to process jobs from Redis queue.

    Args:
        queue_names: List of queue names to listen to (default: ["default"])
        burst: If True, worker exits after processing all jobs (for testing)
    """
    queue_names = queue_names or ["default"]

    logger.info(f"Starting RQ worker for queues: {queue_names}")
    logger.info(f"Redis URL: {settings.redis_url}")

    redis_conn = redis.from_url(settings.redis_url)

    # Create queues with connection
    queues = [Queue(name, connection=redis_conn) for name in queue_names]

    worker = Worker(
        queues,
        name=f"worker-{os.getpid()}",
        connection=redis_conn,
    )

    logger.success(f"Worker started: {worker.name}")

    worker.work(burst=burst, with_scheduler=True)


if __name__ == "__main__":
    """
    Start worker from command line:

    python -m document_parser.worker
    """
    start_worker()
