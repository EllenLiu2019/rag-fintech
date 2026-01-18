from typing import Optional, List
from .registry import ParserRegistry
from .base import BaseParser
from llama_index.core.schema import Document
from pydantic import BaseModel, Field
from common import get_logger

logger = get_logger(__name__)


class ParseResult(BaseModel):
    documents: List[Document] = Field(description="Parsed documents")
    job_id: str = Field(description="Job ID")
    content_type: str = Field(description="Content type", default="application/pdf")


class ContentParser:
    """文本解析器 - 工厂类"""

    def __init__(self):
        self.registry = ParserRegistry()
        self._auto_register()

    def _auto_register(self):
        """自动注册所有解析器"""
        # 延迟导入避免循环依赖
        from .text_parser import TextParser
        from .pdf_parser import PDFParser

        self.registry.register(TextParser)
        self.registry.register(PDFParser)
        # 未来可以自动发现并注册

    def parse(self, contents: bytes, filename: str, content_type: Optional[str] = None) -> ParseResult:
        logger.info("starting to extract file text content...")

        parser = self.registry.get_parser(filename, content_type)

        if not parser:
            raise ValueError(f"unsupported file type: {filename} (content_type: {content_type})")

        try:
            parse_result = parser.parse_content(contents, filename, content_type)
            logger.info(f"length: {len(parse_result.documents)} pages")
            return parse_result
        except Exception as e:
            logger.error(f"file parsing failed [{filename}]: {str(e)}")
            raise

    def register_parser(self, parser_class: type[BaseParser]):
        """动态注册新的解析器"""
        self.registry.register(parser_class)
