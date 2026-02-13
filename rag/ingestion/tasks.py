import asyncio
import uuid
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Callable, List, Optional

from repository.cache import redis_client
from common import get_logger
from common.exceptions import IngestionError
from common.error_codes import ErrorCodes

logger = get_logger(__name__)


class IngestionJob(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="job_id")
    doc_id: str = Field(default="", description="rdb document id")
    step: int = Field(default=0, description="step")
    message: str = Field(default="", description="message")
    status: str = Field(default="pending", description="status")
    created_at: Optional[datetime] = Field(default=None, description="created_at")
    started_at: Optional[datetime] = Field(default=None, description="started_at")
    ended_at: Optional[datetime] = Field(default=None, description="ended_at")
    error: Optional[str] = Field(default=None, description="error")
    result: Optional[str] = Field(default=None, description="result")


class BatchIngestionJob(BaseModel):
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="batch_id")
    jobs: List[IngestionJob] = Field(default_factory=list, description="individual jobs")
    total: int = Field(default=0, description="total number of files")
    accepted: int = Field(default=0, description="number of successfully enqueued files")
    failed: int = Field(default=0, description="number of files that failed to enqueue")


def enqueue_task(func: Callable, *args, **kwargs) -> IngestionJob:
    job_timeout = kwargs.pop("job_timeout", 600)
    # Extract document_id for job tracking (don't pass to the actual function)
    doc_id = kwargs.get("document_id", None)

    try:
        if not redis_client.redis_enabled:
            raise IngestionError(message="Redis is disabled, cannot enqueue task", code=ErrorCodes.S_INGESTION_006)
        job = redis_client.enqueue(
            func,
            *args,
            job_timeout=job_timeout,
            **kwargs,
        )
        logger.info(f"Queued task {func.__name__} with job id {job.id}")
        # Only pass doc_id if it exists, otherwise let Pydantic use default value
        if doc_id is not None:
            ingestion_job = IngestionJob(job_id=job.id, doc_id=doc_id)
        else:
            ingestion_job = IngestionJob(job_id=job.id)
        return ingestion_job
    except Exception as e:
        logger.error(f"Failed to enqueue task {func.__name__}: {e}")
        raise IngestionError(message=f"Failed to enqueue task {func.__name__}: {e}", code=ErrorCodes.S_INGESTION_005)


async def get_task(job_id: str) -> IngestionJob:
    """
    Get task status by job ID.

    Args:
        job_id: Job ID

    Returns:
        IngestionJob with current status
    """
    try:
        job = await asyncio.to_thread(redis_client.get_job, job_id)
        if job is None:
            logger.warning(f"Job {job_id} not found")
            return IngestionJob(job_id=job_id, status="not_found")

        ingestion_job = IngestionJob(
            job_id=job_id,
            status=job.get_status(),
            created_at=job.created_at,
            started_at=job.started_at,
            ended_at=job.ended_at,
        )

        # Get progress if available
        try:
            progress_data = await asyncio.to_thread(redis_client.get_progress, job_id)
            logger.debug(f"Raw progress_data for job {job_id}: {progress_data}")
            if progress_data:
                # Redis returns bytes when decode_responses=False
                step_value = progress_data.get(b"step")
                message_value = progress_data.get(b"message")
                logger.debug(f"Extracted step={step_value}, message={message_value}")
                if step_value:
                    ingestion_job.step = int(step_value.decode())
                if message_value:
                    ingestion_job.message = message_value.decode()
        except Exception as e:
            logger.error(f"Failed to get progress for job {job_id}: {e}", exc_info=True)

        if job.is_failed:
            ingestion_job.error = str(job.exc_info) if job.exc_info else "Unknown error"
            ingestion_job.status = "failed"
        elif job.is_finished:
            ingestion_job.result = "Document processed successfully"
            ingestion_job.status = "finished"

        return ingestion_job
    except Exception as e:
        logger.error(f"Failed to get task {job_id}: {e}", exc_info=True)
        # Return error status instead of raising exception for better UX
        return IngestionJob(job_id=job_id, status="error", error=str(e))


def update_progress(job_id: str, step: int, message: str):
    """
    Update job progress in Redis.

    This function silently fails if progress update fails, as it should not
    interrupt the main ingestion process.
    """
    try:
        redis_client.update_progress(job_id, step, message)
        logger.info(f"Updated progress for job {job_id}: step={step}, message={message}")
    except Exception as e:
        # Log but don't raise - progress update failure shouldn't stop ingestion
        logger.error(f"Failed to update job progress {job_id}: {e}", exc_info=True)
