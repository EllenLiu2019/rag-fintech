import os
import sys
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# 添加项目根目录到 Python 路径（必须在导入 common 模块之前）
project_root = os.path.join(os.path.dirname(__file__), '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入配置
from api.config import settings

# 导入通用模块
from common.log_utils import (
    init_root_logger,
    get_logger
)
from common.log_middleware import setup_request_logging_middleware

# 导入文件解析器
from service.parser import extract_content

# 初始化日志系统
init_root_logger()
logger = get_logger(__name__)

# 创建 FastAPI 应用（使用配置中的元数据）
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

# 存储上传的文件内容（实际应用中应使用数据库）
uploaded_files = {}

@app.post("/api/process")
async def upload_file(file: UploadFile = File(...)):
    """
    处理文件上传
    """
    try:
        logger.info(f"received file uploading request, file_name: {file.filename}")

        contents = await file.read()
        
        logger.info(f"   file size: {len(contents)} bytes")
        
        # 提取文件文本内容
        try:
            text_content = extract_content(contents, file.filename, file.content_type)
        except ValueError as e:
            # 处理不支持的文件类型或解析错误
            logger.error(f"file parsing failed: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={"message": f"file parsing failed: {str(e)}"}
            )

        # 保存文件信息（实际应用中应保存到数据库）
        uploaded_files[file.filename] = {
            "filename": file.filename,
            "size": len(contents),
            "content_type": file.content_type,
            "content": text_content  # 确保是字符串
        }
        
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

@app.get("/api/file-content")
async def get_file_content(filename: str = Query(..., description="文件名")):
    """
    获取上传文件的内容
    """
    try:
        logger.info(f"received file content request, "
                    f"filename: {filename}; "
                    f"stored files: {list(uploaded_files.keys())}")
        
        if filename not in uploaded_files:
            logger.warning(f"file '{filename}' not found")
            return JSONResponse(
                status_code=404,
                content={"message": f"文件 '{filename}' 未找到"}
            )
        
        file_info = uploaded_files[filename]
        logger.info(f"file found: {file_info['filename']}; "
                    f"size: {file_info['size']} bytes; "
                    f"content_type: {file_info['content_type']}")
        
        content = file_info["content"]
        
        logger.info(f"   content length: {len(content)} characters")
        
        return JSONResponse(
            status_code=200,
            content={
                "filename": file_info["filename"],
                "content": content,  # 确保是字符串
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