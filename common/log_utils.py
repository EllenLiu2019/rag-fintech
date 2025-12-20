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

# third party libraries logger names and default log levels
THIRD_PARTY_LOGGERS = {
    "pymilvus": logging.WARNING,
    "transformers": logging.WARNING,
    "transformers.modeling_utils": logging.ERROR,
    "accelerate": logging.WARNING,
    "httpx": logging.WARNING,
    "httpcore": logging.WARNING,
    "urllib3": logging.WARNING,
    "asyncio": logging.WARNING,
    "filelock": logging.WARNING,
}

PRINT_REDIRECT_LIBRARIES = {
    "llama_cloud_services": "llama_cloud_services",
    "llama_index": "llama_index",
}


class RequestIDFormatter(logging.Formatter):
    """自定义日志格式化器，自动从上下文获取 request_id"""

    def format(self, record):
        # 从上下文变量获取 request_id
        record.request_id = request_id_var.get()
        return super().format(record)


def init_root_logger(
    level: Union[int, str] = logging.INFO,
    format_str: Optional[str] = None,
    capture_warnings: bool = True,
    configure_third_party: bool = True,
) -> None:
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
        format_str = "%(asctime)s - [%(request_id)s] - %(name)s - %(levelname)s - %(message)s"

    formatter = RequestIDFormatter(fmt=format_str, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # redirect Python warnings to logging system
    if capture_warnings:
        logging.captureWarnings(True)
        # warnings will be sent to 'py.warnings' logger
        warnings_logger = logging.getLogger("py.warnings")
        warnings_logger.setLevel(logging.WARNING)

    # configure third party libraries log levels
    if configure_third_party:
        _configure_transformers_logging()
        _configure_third_party_loggers()


def _configure_third_party_loggers() -> None:
    for logger_name, log_level in THIRD_PARTY_LOGGERS.items():
        third_party_logger = logging.getLogger(logger_name)
        third_party_logger.setLevel(log_level)
        # 移除第三方库自己的 handler，让它使用 root logger
        third_party_logger.handlers.clear()
        # 确保日志传播到 root logger
        third_party_logger.propagate = True

    # 特殊处理：重定向使用 print() 的第三方库输出到 logging
    _redirect_third_party_prints()


def _redirect_third_party_prints() -> None:
    """
    通过替换内置 print 函数，将使用 print() 的第三方库输出重定向到 logging。

    支持多个库，通过调用栈自动判断来源并路由到对应的 logger。
    """
    try:
        import builtins
        import inspect

        # 为每个需要重定向的库创建 logger
        loggers = {}
        for lib_identifier, logger_name in PRINT_REDIRECT_LIBRARIES.items():
            lib_logger = logging.getLogger(logger_name)
            lib_logger.setLevel(logging.INFO)
            lib_logger.propagate = True
            loggers[lib_identifier] = lib_logger

        # 保存原始的 print 函数
        _original_print = builtins.print

        def _logging_print(*args, **kwargs):
            """替换的 print 函数，自动判断来源并重定向到对应的 logging"""
            try:
                stack = inspect.stack()

                # 检查调用栈，找出是否来自需要重定向的库
                matched_lib = None
                for lib_identifier in PRINT_REDIRECT_LIBRARIES.keys():
                    if any(
                        lib_identifier in str(frame_info.filename) for frame_info in stack[1:8]  # 检查调用栈的前几层
                    ):
                        matched_lib = lib_identifier
                        break

                if matched_lib:
                    # 重定向到对应的 logger
                    message = " ".join(str(arg) for arg in args)
                    if message.strip():
                        loggers[matched_lib].info(message)
                else:
                    # 非需要重定向的库的调用，使用原始 print
                    _original_print(*args, **kwargs)
            except Exception:
                # 如果检查失败，回退到原始 print
                _original_print(*args, **kwargs)

        # 替换内置的 print 函数
        builtins.print = _logging_print

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to redirect third-party prints: {e}")


def _configure_transformers_logging() -> None:
    try:
        import transformers

        # 禁用 transformers 的默认 handler，让它使用 root logger
        transformers.logging.disable_default_handler()
        # 设置日志级别
        transformers.logging.set_verbosity_warning()
        # 重置格式，使用我们的格式
        transformers.logging.reset_format()

        # 确保 transformers 的所有 logger 都使用 root logger
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.handlers.clear()
        transformers_logger.propagate = True
    except ImportError:
        pass  # transformers not installed, skip


def set_third_party_log_level(logger_name: str, level: Union[int, str]) -> None:
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger(logger_name).setLevel(level)


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
