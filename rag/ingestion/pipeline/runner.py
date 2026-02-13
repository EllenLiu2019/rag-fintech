from fastapi import UploadFile
from typing import Callable, List
import asyncio

from rag.persistence import PersistentService
from common import get_logger
from rag.ingestion.pipeline.pipeline_factory import PipelineFactory
from rag.entity import DocumentType
from rag.ingestion.tasks import enqueue_task, update_progress, BatchIngestionJob

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

    async def batch(self, files: List[UploadFile], doc_type: DocumentType, **kwargs) -> BatchIngestionJob:
        """
        Batch process multiple files. Each file is uploaded and enqueued independently.
        Files are uploaded concurrently, then enqueued to RQ for parallel processing by workers.
        """
        batch_job = BatchIngestionJob(total=len(files))

        # Upload all files concurrently
        upload_tasks = [PersistentService.upload_file(f, doc_type.value) for f in files]
        upload_results = await asyncio.gather(*upload_tasks, return_exceptions=True)

        # Enqueue each successfully uploaded file
        for file, result in zip(files, upload_results):
            if isinstance(result, Exception):
                logger.error(f"Failed to upload file {file.filename}: {result}")
                batch_job.failed += 1
                batch_job.jobs.append(_failed_job(file.filename, str(result)))
                continue

            rdb_document = result
            try:
                job = enqueue_task(
                    pipeline,
                    file.filename,
                    file.content_type,
                    doc_type,
                    update_progress,
                    rdb_document_id=rdb_document.id,
                    document_id=rdb_document.document_id,
                    **kwargs,
                )
                batch_job.jobs.append(job)
                batch_job.accepted += 1
                logger.info(f"Enqueued batch job for {file.filename}: {job.job_id}")
            except Exception as e:
                logger.error(f"Failed to enqueue {file.filename}: {e}")
                batch_job.failed += 1
                batch_job.jobs.append(_failed_job(file.filename, str(e)))

        logger.info(
            f"Batch {batch_job.batch_id}: {batch_job.accepted}/{batch_job.total} files enqueued, "
            f"{batch_job.failed} failed"
        )
        return batch_job


def _failed_job(filename: str, error: str):
    """Create a failed IngestionJob for batch error reporting."""
    from rag.ingestion.tasks import IngestionJob
    return IngestionJob(status="failed", error=f"[{filename}] {error}")


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
