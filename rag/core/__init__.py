"""
Core shared services used across multiple RAG stages.
"""

from .embedding_service import embedder
from .doc_service import DocumentService

__all__ = ["embedder", "DocumentService"]
