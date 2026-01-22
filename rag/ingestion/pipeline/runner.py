from fastapi import UploadFile
from typing import Callable
import asyncio

from rag.persistence import PersistentService
from common import get_logger
from rag.ingestion.pipeline.pipeline_factory import PipelineFactory
from rag.entity import DocumentType
from rag.ingestion.tasks import enqueue_task, update_progress

logger = get_logger(__name__)


class PipelineRunner:

    async def __call__(self, file: UploadFile, doc_type: DocumentType, **kwargs):

        rdb_document = await PersistentService.upload_file(file, doc_type.value)

        # Enqueue task - RQ will generate job_id
        # Use module-level function to avoid pickle issues with instance methods
        job = enqueue_task(
            pipeline,  # Synchronous wrapper for RQ worker
            file.filename,
            file.content_type,
            doc_type,
            update_progress,
            rdb_document_id=rdb_document.id,  # Pass as keyword argument
            document_id=rdb_document.document_id,
            **kwargs,
        )

        logger.info(f"Enqueued document ingestion job: {job.job_id}")
        return job


def pipeline(
    filename: str,
    content_type: str,
    doc_type: DocumentType,
    callback: Callable[[str, int, str], None] = None,
    **kwargs,
):
    """
    Synchronous pipeline function for RQ worker.

    Args:
        filename: File name
        content_type: Content type
        doc_type: Document type (default: POLICY)
        callback: Progress update callback function
        **kwargs: Additional arguments (e.g., rdb_document_id)
    """
    pipeline = PipelineFactory.create(doc_type)
    asyncio.run(pipeline.process(filename, content_type, callback, **kwargs))


def _create_runner() -> PipelineRunner:
    return PipelineRunner()


pipeline_runner = _create_runner()
