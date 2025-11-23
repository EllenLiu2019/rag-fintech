"""文件解析器模块"""
from typing import List
from llama_index.core.schema import Document
from .parser import ContentParser
from .base import BaseParser
from .registry import ParserRegistry

# 创建全局实例
_parser = None

def get_parser() -> ContentParser:
    """获取文本解析器单例"""
    global _parser
    if _parser is None:
        _parser = ContentParser()
    return _parser

def parse_content(contents: bytes, filename: str, content_type: str = None) -> List[Document]:
    """便捷函数：解析文件文本"""
    return get_parser().extract(contents, filename, content_type)

__all__ = [
    'ContentParser',
    'BaseParser',
    'ParserRegistry',
    'get_parser',
    'parse_content',
]
