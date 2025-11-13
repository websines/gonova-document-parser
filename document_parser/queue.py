"""Job queue system with Redis persistence.

Provides reliable job storage and processing queue for document processing.
Jobs survive server restarts and failures.
"""

import json
import time
from typing import Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

try:
    import redis
    from redis import Redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning(
        "Redis not installed. Job persistence disabled. "
        "Install with: pip install redis"
    )


class JobStatus:
    """Job status constants."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    """Job model."""

    job_id: str
    filename: str
    status: str
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    accuracy_mode: str
    output_format: str
    vqa_questions: Optional[List[str]] = None
    extract_signatures: Optional[bool] = None
    metadata: Optional[Dict] = None  # For additional params like generate_embeddings
    result: Optional[Dict] = None
    error: Optional[str] = None
    retry_count: int = 0


class JobQueue:
    """
    Persistent job queue using Redis.

    Features:
    - Job persistence (survives server restarts)
    - FIFO queue for processing order
    - Job retry logic
    - Status tracking
    - TTL for completed jobs

    If Redis is not available, falls back to in-memory storage.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        job_ttl: int = 86400,  # 24 hours
        max_retries: int = 3,
    ):
        """
        Initialize job queue.

        Args:
            redis_url: Redis connection URL
            job_ttl: TTL for completed jobs (seconds)
            max_retries: Maximum retry attempts
        """
        self.job_ttl = job_ttl
        self.max_retries = max_retries

        # Try to connect to Redis
        if REDIS_AVAILABLE:
            try:
                self.redis: Optional[Redis] = redis.from_url(
                    redis_url, decode_responses=True
                )
                self.redis.ping()
                logger.success(f"Connected to Redis at {redis_url}")
                self.use_redis = True
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                logger.warning("Falling back to in-memory storage")
                self.redis = None
                self.use_redis = False
        else:
            self.redis = None
            self.use_redis = False

        # Fallback in-memory storage
        if not self.use_redis:
            self._jobs: Dict[str, Job] = {}
            self._queue: List[str] = []

    def _job_key(self, job_id: str) -> str:
        """Get Redis key for job."""
        return f"job:{job_id}"

    def add_job(self, job: Job) -> bool:
        """
        Add job to queue.

        Args:
            job: Job to add

        Returns:
            True if added successfully
        """
        try:
            if self.use_redis:
                # Store job data
                self.redis.set(
                    self._job_key(job.job_id),
                    job.model_dump_json(),
                    ex=self.job_ttl,
                )

                # Add to processing queue
                self.redis.lpush("job_queue", job.job_id)

                logger.info(f"Job {job.job_id} added to Redis queue")
            else:
                # In-memory storage
                self._jobs[job.job_id] = job
                self._queue.append(job.job_id)

                logger.info(f"Job {job.job_id} added to memory queue")

            return True

        except Exception as e:
            logger.error(f"Failed to add job {job.job_id}: {e}")
            return False

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job if found, None otherwise
        """
        try:
            if self.use_redis:
                data = self.redis.get(self._job_key(job_id))
                if data:
                    return Job.model_validate_json(data)
            else:
                return self._jobs.get(job_id)

        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")

        return None

    def update_job(self, job: Job) -> bool:
        """
        Update job status and data.

        Args:
            job: Updated job

        Returns:
            True if updated successfully
        """
        try:
            if self.use_redis:
                # Update with TTL based on status
                ttl = self.job_ttl if job.status == JobStatus.COMPLETED else None
                self.redis.set(
                    self._job_key(job.job_id),
                    job.model_dump_json(),
                    ex=ttl,
                )
            else:
                self._jobs[job.job_id] = job

            return True

        except Exception as e:
            logger.error(f"Failed to update job {job.job_id}: {e}")
            return False

    def get_next_job(self) -> Optional[Job]:
        """
        Get next job from queue.

        Returns:
            Next job to process, or None if queue is empty
        """
        try:
            if self.use_redis:
                # Pop from queue (blocking for 1 second)
                result = self.redis.brpop("job_queue", timeout=1)
                if result:
                    _, job_id = result
                    return self.get_job(job_id)
            else:
                if self._queue:
                    job_id = self._queue.pop(0)
                    return self._jobs.get(job_id)

        except Exception as e:
            logger.error(f"Failed to get next job: {e}")

        return None

    def list_jobs(
        self, status: Optional[str] = None, limit: int = 100
    ) -> List[Job]:
        """
        List jobs with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum number of jobs to return

        Returns:
            List of jobs
        """
        jobs = []

        try:
            if self.use_redis:
                # Scan all job keys
                for key in self.redis.scan_iter("job:*", count=limit):
                    data = self.redis.get(key)
                    if data:
                        job = Job.model_validate_json(data)
                        if status is None or job.status == status:
                            jobs.append(job)
            else:
                jobs = list(self._jobs.values())
                if status:
                    jobs = [j for j in jobs if j.status == status]

            # Sort by creation time (newest first)
            jobs.sort(key=lambda x: x.created_at, reverse=True)

            return jobs[:limit]

        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []

    def delete_job(self, job_id: str) -> bool:
        """
        Delete job.

        Args:
            job_id: Job ID

        Returns:
            True if deleted successfully
        """
        try:
            if self.use_redis:
                self.redis.delete(self._job_key(job_id))
            else:
                if job_id in self._jobs:
                    del self._jobs[job_id]

            return True

        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False

    def get_queue_length(self) -> int:
        """
        Get number of jobs in queue.

        Returns:
            Queue length
        """
        try:
            if self.use_redis:
                return self.redis.llen("job_queue")
            else:
                return len(self._queue)
        except Exception as e:
            logger.error(f"Failed to get queue length: {e}")
            return 0

    def get_stats(self) -> Dict:
        """
        Get queue statistics.

        Returns:
            Dict with queue stats
        """
        try:
            all_jobs = self.list_jobs(limit=1000)

            stats = {
                "total_jobs": len(all_jobs),
                "pending": len([j for j in all_jobs if j.status == JobStatus.PENDING]),
                "processing": len(
                    [j for j in all_jobs if j.status == JobStatus.PROCESSING]
                ),
                "completed": len(
                    [j for j in all_jobs if j.status == JobStatus.COMPLETED]
                ),
                "failed": len([j for j in all_jobs if j.status == JobStatus.FAILED]),
                "queue_length": self.get_queue_length(),
                "using_redis": self.use_redis,
            }

            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        """
        Cleanup old completed/failed jobs.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of jobs deleted
        """
        deleted = 0
        cutoff = time.time() - (max_age_hours * 3600)

        try:
            if self.use_redis:
                # Redis handles TTL automatically, so this is optional
                for key in self.redis.scan_iter("job:*"):
                    data = self.redis.get(key)
                    if data:
                        job = Job.model_validate_json(data)
                        if (
                            job.status in [JobStatus.COMPLETED, JobStatus.FAILED]
                            and job.completed_at
                            and job.completed_at < cutoff
                        ):
                            self.redis.delete(key)
                            deleted += 1
            else:
                job_ids_to_delete = []
                for job_id, job in self._jobs.items():
                    if (
                        job.status in [JobStatus.COMPLETED, JobStatus.FAILED]
                        and job.completed_at
                        and job.completed_at < cutoff
                    ):
                        job_ids_to_delete.append(job_id)

                for job_id in job_ids_to_delete:
                    del self._jobs[job_id]
                    deleted += 1

            logger.info(f"Cleaned up {deleted} old jobs")
            return deleted

        except Exception as e:
            logger.error(f"Failed to cleanup old jobs: {e}")
            return 0
