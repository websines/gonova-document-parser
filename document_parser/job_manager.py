"""Job manager for Redis-based state persistence and result storage."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import redis
from loguru import logger
from pydantic import BaseModel

from document_parser.config import settings


class JobStatus(BaseModel):
    """Job status model for tracking processing state."""

    job_id: str
    status: str  # queued, processing, completed, failed
    filename: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Dict = {}  # {current_page: int, total_pages: int, model_routing: dict}
    error: Optional[str] = None
    result_path: Optional[str] = None
    metadata: Dict = {}


class JobManager:
    """
    Manages job state persistence and result storage in Redis.

    Features:
    - Job status tracking (queued → processing → completed/failed)
    - Progress updates (current page, routing stats)
    - Result storage (markdown output)
    - TTL-based cleanup (auto-delete old jobs)
    """

    def __init__(self, redis_url: str = None, ttl_hours: int = 24):
        """
        Initialize job manager.

        Args:
            redis_url: Redis connection URL (default: from settings)
            ttl_hours: Time-to-live for completed jobs (default: 24 hours)
        """
        self.redis_url = redis_url or settings.redis_url
        self.ttl_seconds = ttl_hours * 3600
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        logger.info(f"JobManager initialized with Redis: {self.redis_url}")

    def _job_key(self, job_id: str) -> str:
        """Get Redis key for job status."""
        return f"job:{job_id}:status"

    def _result_key(self, job_id: str) -> str:
        """Get Redis key for job result."""
        return f"job:{job_id}:result"

    def create_job(self, job_id: str, filename: str, metadata: Dict = None) -> JobStatus:
        """
        Create a new job with 'queued' status.

        Args:
            job_id: Unique job identifier
            filename: Original PDF filename
            metadata: Optional metadata (accuracy_mode, output_format, etc.)

        Returns:
            JobStatus object
        """
        job = JobStatus(
            job_id=job_id,
            status="queued",
            filename=filename,
            created_at=datetime.utcnow().isoformat(),
            metadata=metadata or {},
        )

        # Store in Redis with TTL
        self.redis_client.setex(
            self._job_key(job_id),
            self.ttl_seconds,
            job.model_dump_json(),
        )

        logger.info(f"Created job {job_id}: {filename}")
        return job

    def get_job(self, job_id: str) -> Optional[JobStatus]:
        """
        Get job status.

        Args:
            job_id: Job identifier

        Returns:
            JobStatus object or None if not found
        """
        data = self.redis_client.get(self._job_key(job_id))
        if not data:
            return None

        return JobStatus.model_validate_json(data)

    def update_status(
        self,
        job_id: str,
        status: str,
        progress: Dict = None,
        error: str = None,
    ):
        """
        Update job status and progress.

        Args:
            job_id: Job identifier
            status: New status (processing, completed, failed)
            progress: Progress dict {current_page, total_pages, routing}
            error: Error message if failed
        """
        job = self.get_job(job_id)
        if not job:
            logger.error(f"Job {job_id} not found for status update")
            return

        job.status = status

        if status == "processing" and not job.started_at:
            job.started_at = datetime.utcnow().isoformat()

        if status in ("completed", "failed"):
            job.completed_at = datetime.utcnow().isoformat()

        if progress:
            job.progress = progress

        if error:
            job.error = error

        # Update in Redis with TTL
        self.redis_client.setex(
            self._job_key(job_id),
            self.ttl_seconds,
            job.model_dump_json(),
        )

        logger.info(f"Updated job {job_id}: status={status}")

    def store_result(self, job_id: str, markdown_output: str):
        """
        Store processing result (markdown output) in Redis.

        Args:
            job_id: Job identifier
            markdown_output: Processed markdown content
        """
        # Store result with TTL
        self.redis_client.setex(
            self._result_key(job_id),
            self.ttl_seconds,
            markdown_output,
        )

        # Update job with result availability
        job = self.get_job(job_id)
        if job:
            job.result_path = f"/v1/jobs/{job_id}/result"
            self.redis_client.setex(
                self._job_key(job_id),
                self.ttl_seconds,
                job.model_dump_json(),
            )

        logger.info(f"Stored result for job {job_id} ({len(markdown_output)} bytes)")

    def get_result(self, job_id: str) -> Optional[str]:
        """
        Get processing result (markdown output).

        Args:
            job_id: Job identifier

        Returns:
            Markdown content or None if not found
        """
        return self.redis_client.get(self._result_key(job_id))

    def delete_job(self, job_id: str):
        """
        Delete job and its result from Redis.

        Args:
            job_id: Job identifier
        """
        self.redis_client.delete(self._job_key(job_id))
        self.redis_client.delete(self._result_key(job_id))
        logger.info(f"Deleted job {job_id}")

    def list_jobs(self, limit: int = 100) -> list[JobStatus]:
        """
        List recent jobs.

        Args:
            limit: Maximum number of jobs to return

        Returns:
            List of JobStatus objects
        """
        pattern = "job:*:status"
        keys = list(self.redis_client.scan_iter(match=pattern, count=limit))[:limit]

        jobs = []
        for key in keys:
            data = self.redis_client.get(key)
            if data:
                try:
                    jobs.append(JobStatus.model_validate_json(data))
                except Exception as e:
                    logger.error(f"Error parsing job from {key}: {e}")

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def cleanup_old_jobs(self, days: int = 7):
        """
        Manually cleanup jobs older than specified days.

        Args:
            days: Delete jobs older than this many days
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        deleted = 0

        for job in self.list_jobs(limit=1000):
            created_at = datetime.fromisoformat(job.created_at)
            if created_at < cutoff:
                self.delete_job(job.job_id)
                deleted += 1

        logger.info(f"Cleaned up {deleted} old jobs (older than {days} days)")
