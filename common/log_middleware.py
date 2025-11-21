"""
通用中间件模块
包含请求日志、请求 ID 追踪等中间件
"""
import time
from fastapi import Request
from typing import Callable

from .log_utils import (
    get_logger,
    set_request_id,
    generate_request_id
)

logger = get_logger(__name__)


async def request_logging_middleware(request: Request, call_next: Callable):
    """
    请求日志中间件
    为每个请求生成 UUID 并记录请求和响应信息
    
    Args:
        request: FastAPI 请求对象
        call_next: 下一个中间件或路由处理函数
        
    Returns:
        Response: HTTP 响应对象
    """
    # 生成并设置请求 ID
    request_id = generate_request_id()
    request.state.request_id = request_id
    set_request_id(request_id)
    
    start_time = time.time()
    
    # 记录请求信息
    logger.info(f"🌐 {request.method} {request.url.path}")
    if request.query_params:
        logger.info(f"   query params: {dict(request.query_params)}")
    
    # 处理请求（异步调用）
    response = await call_next(request)
    
    # 在响应头中添加请求 ID，方便前端追踪
    response.headers["X-Request-ID"] = request_id
    
    # 记录响应信息
    process_time = time.time() - start_time
    logger.info(f"   response: {response.status_code} (elapsed: {process_time:.3f}s)")
    
    return response


def setup_request_logging_middleware(app):
    """
    设置请求日志中间件到 FastAPI 应用
    
    Args:
        app: FastAPI 应用实例
    """
    @app.middleware("http")
    async def middleware(request: Request, call_next: Callable):
        return await request_logging_middleware(request, call_next)

