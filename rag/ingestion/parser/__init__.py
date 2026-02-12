from typing import List
from llama_index.core.schema import Document
from .parser import ContentParser
from .base import BaseParser
from .registry import ParserRegistry
from .parser import ParseResult

# Create a global instance
_parser = None


def get_parser() -> ContentParser:
    """Get the text parser singleton"""
    global _parser
    if _parser is None:
        _parser = ContentParser()
    return _parser


def parse_content(contents: bytes, filename: str, content_type: str = None) -> List[Document]:
    return get_parser().parse(contents, filename, content_type)


async def aparse_content(contents: bytes, filename: str, content_type: str = None) -> ParseResult:
    """Async version of parse_content."""
    return await get_parser().aparse(contents, filename, content_type)


__all__ = [
    "ContentParser",
    "BaseParser",
    "ParserRegistry",
    "get_parser",
    "parse_content",
    "aparse_content",
    "ParseResult",
]
