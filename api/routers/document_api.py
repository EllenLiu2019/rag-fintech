from fastapi import APIRouter, File, UploadFile, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from common import get_logger
from common.exceptions import NotFoundError
from common.error_codes import ErrorCodes
from repository.s3 import s3_client
from rag.ingestion import get_task, ingestion_pipeline
from rag.marshaller import to_json

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Document"],
    responses={404: {"description": "Not found"}},
)


class UploadedDoc(BaseModel):
    file_name: str
    task_id: str
    content_type: str
    message: str


@router.post("/process")
async def upload_file(
    file: UploadFile = File(...),
):
    """
    Upload and process a document file.

    - **file**: The document file to upload (PDF, TXT, etc.)
    """
    logger.info(f"Received file upload request: {file.filename}")

    task_id = await ingestion_pipeline(file)

    uploaded_doc = UploadedDoc(
        file_name=file.filename,
        task_id=task_id,
        content_type=file.content_type,
        message=f"File '{file.filename}' accepted",
    )

    return JSONResponse(
        status_code=202,
        content=to_json(uploaded_doc, indent=4, exclude_none=True),
    )


@router.get("/process/{job_id}")
async def get_process_status(job_id: str):
    """Get status of a document processing job."""
    ingestion_job = get_task(job_id)
    return JSONResponse(content=to_json(ingestion_job, indent=4, exclude_none=True))


@router.get("/file-original")
async def get_original_file(filename: str = Query(..., description="文件名")):
    """
    Get original uploaded file (PDF, TXT, etc.)

    - **filename**: Name of the file to retrieve
    """
    logger.info(f"Received original file request: {filename}")

    # Load original file from disk
    file_contents = s3_client.load_original_file(filename)
    if file_contents is None:
        raise NotFoundError(
            message=f"文件 '{filename}' 未找到",
            code=ErrorCodes.A_NOTFOUND_001,
            details={"filename": filename},
        )

    # Get file info to determine content_type
    file_info = s3_client.load_file_info(filename)
    content_type = (
        file_info.get("content_type", "application/octet-stream") if file_info else "application/octet-stream"
    )

    logger.info(f"Original file found: {filename}; size: {len(file_contents)} bytes; content_type: {content_type}")

    # Return file content
    return Response(
        content=file_contents,
        media_type=content_type,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Content-Length": str(len(file_contents)),
        },
    )


@router.get("/file-parsed")
async def get_file_parsed(filename: str = Query(..., description="文件名")):
    """
    Get parsed content of uploaded file

    - **filename**: Name of the file to retrieve
    """
    stored_files = s3_client.list_stored_files()
    logger.info(f"Received file content request: filename={filename}, stored_files={stored_files}")

    file_info = s3_client.load_file_info(filename)
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
            "content_type": file_info["content_type"],
        },
    )
