import mimetypes
from typing import List
from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import asyncio

from common import get_logger
from common.exceptions import NotFoundError, ValidationError, RateLimitExceededError
from common.error_codes import ErrorCodes
from repository.s3 import s3_client
from rag.ingestion.pipeline import pipeline_runner
from rag.ingestion import get_task
from rag.entity import DocumentType
from rag.persistence.persistent_service import PersistentService

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Document"],
    responses={404: {"description": "Not found"}},
)


class UploadedDoc(BaseModel):
    task_id: str
    doc_id: str
    file_name: str
    message: str


batch_semaphore = asyncio.Semaphore(3)


@router.post("/process")
async def upload_file(
    file: UploadFile = File(...),
):
    """
    Upload and process a document file.

    - **file**: The document file to upload (PDF, TXT, etc.)
    """
    logger.info(f"Received file upload request: {file.filename}")

    ingestion_job = await pipeline_runner(file, DocumentType.POLICY, graph_enabled=True)

    uploaded_doc = UploadedDoc(
        task_id=ingestion_job.job_id,
        doc_id=ingestion_job.doc_id,
        file_name=file.filename,
        message=f"File '{file.filename}' accepted",
    )

    return JSONResponse(
        status_code=202,
        content=uploaded_doc.model_dump(mode="json", exclude_none=True),
    )


@router.post("/process/batch")
async def upload_files_batch(
    files: List[UploadFile] = File(...),
):
    """
    Upload and process multiple document files in batch.

    - **files**: Multiple document files to upload (PDF, TXT, etc.)

    Each file is uploaded and enqueued independently.
    Returns a batch_id for tracking and individual job status for each file.
    """
    logger.info(f"Received batch upload request: {len(files)} files")

    if len(files) > 10:
        raise ValidationError(
            message="Batch size limit: 10 files",
            code=ErrorCodes.A_VALIDATION_002,
            details={"batch_size": len(files)},
        )
    if batch_semaphore.locked():
        raise RateLimitExceededError(
            message="Too many batch requests in progress",
            code=ErrorCodes.A_RATELIMIT_001,
            details={"batch_semaphore": batch_semaphore._value},
        )

    async with batch_semaphore:
        batch_job = await pipeline_runner.batch(files, DocumentType.POLICY, graph_enabled=True)

    return JSONResponse(status_code=202, content=batch_job.model_dump(mode="json", exclude_none=True))


@router.get("/documents")
async def list_documents(doc_type: str = Query(default="all", description="Filter by doc_type: all, policy, claim")):
    """List all documents, optionally filtered by doc_type."""
    try:
        records = await PersistentService.alist_documents(doc_type)
        results = []
        for r in records:
            row = r.to_dict()
            # Convert datetime for JSON serialization
            if row.get("upload_time") is not None:
                row["upload_time"] = row["upload_time"].isoformat()
            # Add frontend-friendly aliases
            row["doc_id"] = row.get("document_id")
            row["size"] = row.get("file_size")
            row["status"] = row.get("doc_status")
            row["created_at"] = row.get("upload_time")
            # Drop heavy fields not needed for list display
            row.pop("clause_forest", None)
            row.pop("business_data", None)
            row.pop("confidence", None)
            results.append(row)
        # Sort by upload_time descending (newest first)
        results.sort(key=lambda x: x.get("upload_time", ""), reverse=True)
        return JSONResponse(status_code=200, content=results)
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"detail": str(e)})


@router.get("/process/{job_id}")
async def get_process_status(job_id: str):
    """Get status of a document processing job."""
    ingestion_job = await get_task(job_id)
    return JSONResponse(content=ingestion_job.model_dump(mode="json", exclude_none=True))


@router.get("/original-file")
async def get_original_file(
    filename: str = Query(..., description="文件名"),
    doc_id: str = Query(..., description="文档ID"),
    doc_type: str = Query(default="policy", description="文档类型: policy 或 claim"),
):
    """
    Get original uploaded file (PDF, TXT, etc.)
    """
    logger.info(f"Received original file request: filename={filename}, doc_id={doc_id}, doc_type={doc_type}")

    # Validate and convert doc_type
    try:
        document_type = DocumentType(doc_type.lower())
    except ValueError:
        raise NotFoundError(
            message=f"无效的文档类型: {doc_type}",
            code=ErrorCodes.A_NOTFOUND_001,
            details={"doc_type": doc_type},
        )

    # Load original file from disk
    file_contents = s3_client.load_original_file(filename, doc_id, document_type.value)
    if file_contents is None:
        raise NotFoundError(
            message=f"文件 '{filename}' 未找到",
            code=ErrorCodes.A_NOTFOUND_001,
            details={"filename": filename, "doc_id": doc_id},
        )

    # Determine content_type from file extension
    content_type, _ = mimetypes.guess_type(filename)
    if content_type is None:
        content_type = "application/octet-stream"

    logger.info(f"Original file found: {filename}; size: {len(file_contents)} bytes; content_type: {content_type}")

    return Response(
        content=file_contents,
        headers={
            "Content-Type": content_type,
            "Content-Disposition": f'inline; filename="{filename}"',
            "Content-Length": str(len(file_contents)),
        },
    )


@router.get("/parsed-file")
async def get_parsed_file(
    filename: str = Query(..., description="文件名"),
    doc_id: str = Query(..., description="文档ID"),
    doc_type: str = Query(default="policy", description="文档类型: policy 或 claim"),
):
    """
    Get parsed content of uploaded file
    """
    logger.info(f"Received parsed file request: filename={filename}, doc_id={doc_id}, doc_type={doc_type}")

    # Validate and convert doc_type
    try:
        document_type = DocumentType(doc_type.lower())
    except ValueError:
        raise NotFoundError(
            message=f"无效的文档类型: {doc_type}",
            code=ErrorCodes.A_NOTFOUND_001,
            details={"doc_type": doc_type},
        )

    file_info = s3_client.load_parsed_file(filename, doc_id, document_type.value)
    if file_info is None:
        raise NotFoundError(
            message=f"文件 '{filename}' 未找到",
            code=ErrorCodes.A_NOTFOUND_001,
            details={"filename": filename},
        )

    logger.info(
        f"File found: {file_info['filename']}; "
        f"size: {file_info['file_size']} bytes; "
        f"content_type: {file_info['content_type']}"
    )

    pages = file_info["pages"]
    business_data = file_info.get("business_data", {})
    confidence = file_info.get("overall_confidence", 0)
    document_id = file_info.get("document_id")

    logger.info(f"Documents: {len(pages)} pages, document_id: {document_id}")

    return JSONResponse(
        status_code=200,
        content={
            "filename": file_info["filename"],
            "pages": pages,
            "business_data": business_data,
            "confidence": confidence,
            "document_id": document_id,
            "size": file_info["file_size"],
        },
    )
