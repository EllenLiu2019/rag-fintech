from abc import ABC, abstractmethod
from typing import Any

from rag.entity import RagDocument


class BaseSplitter(ABC):
    """
    Abstract base class for document splitters.
    """

    @abstractmethod
    def split_document(self, doc: RagDocument) -> list[dict[str, Any]]:
        """
        Split a RagDocument into a list of chunks (dictionaries).

        Args:
            doc: The RagDocument to split.

        Returns:
            A list of dictionaries, where each dictionary represents a chunk.
            Expected keys in the chunk dict: 'chunk_id', 'text', 'metadata'.
        """
        pass
