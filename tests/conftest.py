"""
pytest 配置文件
提供共享的 fixtures 和配置
"""
import sys
import os

# 添加项目根目录到 Python 路径
project_root = os.path.join(os.path.dirname(__file__), '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 初始化日志系统（测试环境）
from common.log_utils import init_root_logger
init_root_logger()

