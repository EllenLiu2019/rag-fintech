"""文件解析器模块"""
from .extractor import ContentExtractor
from .base import BaseParser
from .registry import ParserRegistry

# 创建全局实例
_extractor = None

def get_extractor() -> ContentExtractor:
    """获取文本提取器单例"""
    global _extractor
    if _extractor is None:
        _extractor = ContentExtractor()
    return _extractor

def extract_content(contents: bytes, filename: str, content_type: str = None) -> str:
    """便捷函数：提取文件文本"""
    return get_extractor().extract(contents, filename, content_type)

__all__ = [
    'ContentExtractor',
    'BaseParser',
    'ParserRegistry',
    'get_extractor',
    'extract_content',
]
