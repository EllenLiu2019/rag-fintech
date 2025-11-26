from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from api.config import settings
from common.log_utils import (
    init_root_logger,
    get_logger
)
from common.log_middleware import setup_request_logging_middleware
from service.parser import parse_content
from service.parser.serializer_deserializer import serialize_documents
from api.db.persist_file import (
    save_file_info,
    load_file_info,
    list_stored_files,
    save_original_file,
    load_original_file
)

init_root_logger()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置 CORS 中间件（使用配置）
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# 🔍 添加请求日志中间件
setup_request_logging_middleware(app)

@app.post("/api/process")
async def upload_file(file: UploadFile = File(...)):
    """
    处理文件上传
    """
    try:
        logger.info(f"received file uploading request, file_name: {file.filename}")

        contents = await file.read()
        
        logger.info(f"file size: {len(contents)} bytes")
        
        # 保存原始文件到磁盘
        if not save_original_file(file.filename, contents):
            logger.warning(f"failed to save original file to disk for '{file.filename}'")
        
        # 解析文件文本内容
        try:
            documents = parse_content(contents, file.filename, file.content_type)
        except ValueError as e:
            # 处理不支持的文件类型或解析错误
            logger.error(f"file parsing failed: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"message": f"file parsing failed: {str(e)}"}
            )

        # 将 Document 对象列表转换为可序列化的格式
        # 使用序列化器统一处理（类似 Java 的 serialize 方法）
        documents_serializable = serialize_documents(documents)

        # 保存文件信息到磁盘（JSON 格式）
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

@app.get("/api/file-parsed")
async def get_file_parsed(filename: str = Query(..., description="文件名")):
    """
    获取上传文件的解析后内容
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
                "documents": documents,  # 确保是字符串
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

@app.get("/api/file-original")
async def get_original_file(filename: str = Query(..., description="文件名")):
    """
    获取上传的原始文件（PDF、TXT 等）
    """
    try:
        logger.info(f"received original file request, filename: {filename}")
        
        # 从磁盘加载原始文件
        file_contents = load_original_file(filename)
        if file_contents is None:
            logger.warning(f"original file '{filename}' not found")
            return JSONResponse(
                status_code=404,
                content={"message": f"文件 '{filename}' 未找到"}
            )
        
        # 获取文件信息以确定 content_type
        file_info = load_file_info(filename)
        content_type = file_info.get("content_type", "application/octet-stream") if file_info else "application/octet-stream"
        
        logger.info(f"original file found: {filename}; size: {len(file_contents)} bytes; content_type: {content_type}")
        
        # 返回文件内容
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