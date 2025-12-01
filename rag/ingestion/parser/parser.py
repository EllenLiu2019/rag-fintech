from typing import Optional, List
from .registry import ParserRegistry
from .base import BaseParser
from llama_index.core.schema import Document

import logging

logger = logging.getLogger(__name__)

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
    
    def parse(self, contents: bytes, filename: str, content_type: Optional[str] = None) -> List[Document]:
        logger.info("starting to extract file text content...")
        
        parser = self.registry.get_parser(filename, content_type)
        
        if not parser:
            raise ValueError(f"unsupported file type: {filename} (content_type: {content_type})")
        
        try:
            documents = parser.parse_content(contents, filename, content_type)
            logger.info(f"length: {len(documents)} pages")
            return documents
        except Exception as e:
            logger.error(f"file parsing failed [{filename}]: {str(e)}")
            raise
    
    def register_parser(self, parser_class: type[BaseParser]):
        """动态注册新的解析器"""
        self.registry.register(parser_class)

