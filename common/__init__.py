"""
通用工具模块
包含日志、工具函数、中间件等通用功能
"""

from .log_utils import (
    init_root_logger,
    get_logger,
    set_request_id,
    get_request_id,
    generate_request_id,
    RequestIDFormatter,
    request_id_var
)

from .log_middleware import (
    setup_request_logging_middleware,
    request_logging_middleware
)

__all__ = [
    # 日志工具
    'init_root_logger',
    'get_logger',
    'set_request_id',
    'get_request_id',
    'generate_request_id',
    'RequestIDFormatter',
    'request_id_var',
    # 中间件
    'setup_request_logging_middleware',
    'request_logging_middleware',
]

