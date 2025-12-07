from fastapi import APIRouter, File, UploadFile, Query, Depends
from fastapi.responses import JSONResponse, Response

from common.log_utils import get_logger
from rag.ingestion.pipeline import IngestionPipeline
from rag.dependencies import get_ingestion_pipeline
from rag.ingestion.file_service import (
    load_file_info,
    list_stored_files,
    load_original_file,
)

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Document"],
    responses={404: {"description": "Not found"}},
)


@router.post("/process")
async def upload_file(
    file: UploadFile = File(...),
    ingestion_pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
):

    try:
        logger.info(f"received file uploading request, file_name: {file.filename}")

        contents = await file.read()

        ingestion_pipeline.handle_document(file.filename, contents, file.content_type)

        return JSONResponse(
            status_code=200,
            content={
                "message": f"文件 '{file.filename}' 上传成功",
                "filename": file.filename,
                "size": len(contents),
                "content_type": file.content_type,
            },
        )
    except Exception as e:
        logger.error(f"file upload failed: {str(e)}")
        import traceback

        logger.error(f"   error stack:\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"message": f"文件上传失败: {str(e)}"})


@router.get("/file-original")
async def get_original_file(filename: str = Query(..., description="文件名")):
    """
    Get original uploaded file (PDF, TXT, etc.)

    - **filename**: Name of the file to retrieve
    """
    try:
        logger.info(f"received original file request, filename: {filename}")

        # Load original file from disk
        file_contents = load_original_file(filename)
        if file_contents is None:
            logger.warning(f"original file '{filename}' not found")
            return JSONResponse(status_code=404, content={"message": f"文件 '{filename}' 未找到"})

        # Get file info to determine content_type
        file_info = load_file_info(filename)
        content_type = (
            file_info.get("content_type", "application/octet-stream") if file_info else "application/octet-stream"
        )

        logger.info(f"original file found: {filename}; size: {len(file_contents)} bytes; content_type: {content_type}")

        # Return file content
        return Response(
            content=file_contents,
            media_type=content_type,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Content-Length": str(len(file_contents)),
            },
        )
    except Exception as e:
        logger.error(f"failed to get original file: {str(e)}")
        import traceback

        logger.error(f"   error stack:\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"message": f"获取原始文件失败: {str(e)}"})


@router.get("/file-parsed")
async def get_file_parsed(filename: str = Query(..., description="文件名")):
    """
    Get parsed content of uploaded file

    - **filename**: Name of the file to retrieve
    """
    try:
        stored_files = list_stored_files()
        logger.info(f"received file content request, " f"filename: {filename}; " f"stored files: {stored_files}")

        file_info = load_file_info(filename)
        if file_info is None:
            logger.warning(f"file '{filename}' not found")
            return JSONResponse(status_code=404, content={"message": f"文件 '{filename}' 未找到"})
        logger.info(
            f"file found: {file_info['filename']}; "
            f"size: {file_info['file_size']} bytes; "
            f"content_type: {file_info['content_type']}"
        )

        pages = file_info["pages"]
        business_data = file_info.get("business_data", {})
        confidence = file_info.get("overall_confidence", 0)
        document_id = file_info.get("document_id")  # Extract document_id

        logger.info(f"documents: {len(pages)} pages, document_id: {document_id}")

        return JSONResponse(
            status_code=200,
            content={
                "filename": file_info["filename"],
                "pages": pages,
                "business_data": business_data,
                "confidence": confidence,
                "document_id": document_id,  # Include document_id in response
                "size": file_info["file_size"],
                "content_type": file_info["content_type"],
            },
        )
    except Exception as e:
        logger.error(f"failed to get file content: {str(e)}")
        import traceback

        logger.error(f"   error stack:\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"message": f"获取文件内容失败: {str(e)}"})
