import asyncio
from abc import ABC, abstractmethod
from typing import Optional, List

from llama_index.core.schema import Document


class BaseParser(ABC):

    supported_extensions: List[str] = []

    supported_mime_types: List[str] = []

    @abstractmethod
    def can_parse(self, filename: str, content_type: Optional[str]) -> bool:
        """Check if the file can be parsed"""
        pass

    @abstractmethod
    def parse_content(self, contents: bytes, filename: str, content_type: Optional[str]) -> List[Document]:
        """Parse the text content"""
        pass

    async def aparse_content(self, contents: bytes, filename: str, content_type: Optional[str]) -> List[Document]:
        """Async version of parse_content. Override for native async support."""
        return await asyncio.to_thread(self.parse_content, contents, filename, content_type)

    def get_metadata(self, contents: bytes, filename: str) -> dict:
        """Get the file metadata (optional)"""
        return {}
