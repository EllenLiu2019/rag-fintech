"""
通用日志工具模块
提供请求 ID 追踪和日志格式化功能
"""

import logging
import uuid
from contextvars import ContextVar
from typing import Optional, Union

# 创建上下文变量来存储请求 ID
request_id_var: ContextVar[str] = ContextVar("request_id", default="N/A")


class RequestIDFormatter(logging.Formatter):
    """自定义日志格式化器，自动从上下文获取 request_id"""

    def format(self, record):
        # 从上下文变量获取 request_id
        record.request_id = request_id_var.get()
        return super().format(record)


def init_root_logger(level: Union[int, str] = logging.INFO, format_str: Optional[str] = None) -> None:
    """
    初始化根日志记录器，配置请求 ID 格式化
    """
    # 配置 root logger，这样所有子 logger 都会使用这个格式
    root_logger = logging.getLogger()

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root_logger.setLevel(level)

    # 清除可能存在的默认 handler
    if root_logger.handlers:
        root_logger.handlers.clear()

    handler = logging.StreamHandler()

    if not format_str:
        format_str = "%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s  - %(message)s"

    formatter = RequestIDFormatter(fmt=format_str, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志记录器

    Args:
        name: 日志记录器名称，通常使用 __name__

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    return logging.getLogger(name)


def set_request_id(request_id: Optional[str] = None) -> str:
    """
    设置当前请求的 ID

    Args:
        request_id: 请求 ID，如果为 None 则自动生成

    Returns:
        str: 设置的请求 ID
    """
    if request_id is None:
        # 生成唯一的 UUID（使用前8位，更简洁易读）
        request_id = str(uuid.uuid4())[:8]

    request_id_var.set(request_id)
    return request_id


def get_request_id() -> str:
    """
    获取当前请求的 ID

    Returns:
        str: 当前请求 ID，如果没有则返回 'N/A'
    """
    return request_id_var.get()


def generate_request_id() -> str:
    """
    生成新的请求 ID（不设置到上下文）

    Returns:
        str: 新生成的请求 ID（前8位）
    """
    return str(uuid.uuid4())[:8]


def log_exception(e: Exception, msg: str = None) -> None:
    """
    记录异常日志的辅助函数

    Args:
        e: 异常对象
        msg: 可选的额外错误消息
    """
    logger = get_logger(__name__)
    if msg:
        logger.error(f"{msg}: {str(e)}", exc_info=True)
    else:
        logger.error(f"An exception occurred: {str(e)}", exc_info=True)
