from .base import BaseParser
from typing import Optional
from langchain_community.document_loaders import TextLoader
from common import get_logger
from rag.ingestion.parser.parser import ParseResult

logger = get_logger(__name__)


class TextParser(BaseParser):
    """Text Parser"""

    supported_extensions = [".txt", ".md", ".markdown"]
    supported_mime_types = ["text/plain", "text/markdown", "text/x-markdown"]

    def can_parse(self, filename: str, content_type: Optional[str]) -> bool:
        """Check if the file can be parsed"""
        # Check if the file extension matches
        ext_match = filename.lower().endswith(tuple(self.supported_extensions))

        # Check if the MIME type matches
        mime_match = False
        if content_type:
            content_type_lower = content_type.lower()
            mime_match = (
                any(mime in content_type_lower for mime in self.supported_mime_types) or "text" in content_type_lower
            )

        return ext_match or mime_match

    def parse_content(self, filename: str, content_type: Optional[str]) -> ParseResult:
        try:
            loader = TextLoader(filename, encoding="utf-8")
            documents = loader.load()
            return ParseResult(documents=documents, job_id="text_parser", content_type=content_type)
        except Exception as e:
            error_msg = f"text file decoding failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
