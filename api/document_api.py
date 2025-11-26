"""
Document API endpoints
Handles file upload, parsing, and retrieval operations
"""
from fastapi import File, UploadFile, Query
from fastapi.responses import JSONResponse, Response

from common.log_utils import get_logger
from service.parser import parse_content
from service.parser.serializer_deserializer import serialize_documents
from api.db.persist_file import (
    save_file_info,
    load_file_info,
    list_stored_files,
    save_original_file,
    load_original_file
)

logger = get_logger(__name__)


async def upload_file(file: UploadFile = File(...)):
    """
    Handle file upload
    """
    try:
        logger.info(f"received file uploading request, file_name: {file.filename}")

        contents = await file.read()
        
        logger.info(f"file size: {len(contents)} bytes")
        
        # Save original file to disk
        if not save_original_file(file.filename, contents):
            logger.warning(f"failed to save original file to disk for '{file.filename}'")
        
        # Parse file content
        try:
            documents = parse_content(contents, file.filename, file.content_type)
        except ValueError as e:
            # Handle unsupported file types or parsing errors
            logger.error(f"file parsing failed: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"message": f"file parsing failed: {str(e)}"}
            )

        # Convert Document objects to serializable format
        documents_serializable = serialize_documents(documents)

        # Save file info to disk (JSON format)
        file_info = {
            "filename": file.filename,
            "size": len(contents),
            "content_type": file.content_type,
            "documents": documents_serializable
        }
        
        if not save_file_info(file.filename, file_info):
            logger.warning(f"failed to save file info to disk for '{file.filename}'")
        
        logger.info(f"file '{file.filename}' uploaded and saved successfully")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"文件 '{file.filename}' 上传成功",
                "filename": file.filename,
                "size": len(contents),
                "content_type": file.content_type
            }
        )
    except Exception as e:
        logger.error(f"file upload failed: {str(e)}")
        logger.error(f"   error type: {type(e).__name__}")
        import traceback
        logger.error(f"   error stack:\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"message": f"文件上传失败: {str(e)}"}
        )


async def get_file_parsed(filename: str = Query(..., description="文件名")):
    """
    Get parsed content of uploaded file
    """
    try:
        stored_files = list_stored_files()
        logger.info(f"received file content request, "
                    f"filename: {filename}; "
                    f"stored files: {stored_files}")
        
        file_info = load_file_info(filename)
        if file_info is None:
            logger.warning(f"file '{filename}' not found")
            return JSONResponse(
                status_code=404,
                content={"message": f"文件 '{filename}' 未找到"}
            )
        logger.info(f"file found: {file_info['filename']}; "
                    f"size: {file_info['size']} bytes; "
                    f"content_type: {file_info['content_type']}")
        
        documents = file_info["documents"]
        
        logger.info(f"documents: {len(documents)} pages")
        
        return JSONResponse(
            status_code=200,
            content={
                "filename": file_info["filename"],
                "documents": documents,
                "size": file_info["size"],
                "content_type": file_info["content_type"]
            }
        )
    except Exception as e:
        logger.error(f"failed to get file content: {str(e)}")
        import traceback
        logger.error(f"   error stack:\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"message": f"获取文件内容失败: {str(e)}"}
        )


async def get_original_file(filename: str = Query(..., description="文件名")):
    """
    Get original uploaded file (PDF, TXT, etc.)
    """
    try:
        logger.info(f"received original file request, filename: {filename}")
        
        # Load original file from disk
        file_contents = load_original_file(filename)
        if file_contents is None:
            logger.warning(f"original file '{filename}' not found")
            return JSONResponse(
                status_code=404,
                content={"message": f"文件 '{filename}' 未找到"}
            )
        
        # Get file info to determine content_type
        file_info = load_file_info(filename)
        content_type = file_info.get("content_type", "application/octet-stream") if file_info else "application/octet-stream"
        
        logger.info(f"original file found: {filename}; size: {len(file_contents)} bytes; content_type: {content_type}")
        
        # Return file content
        return Response(
            content=file_contents,
            media_type=content_type,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Content-Length": str(len(file_contents))
            }
        )
    except Exception as e:
        logger.error(f"failed to get original file: {str(e)}")
        import traceback
        logger.error(f"   error stack:\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"message": f"获取原始文件失败: {str(e)}"}
        )

