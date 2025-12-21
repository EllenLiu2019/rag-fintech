"""
Document ingestion pipeline.

This module handles the complete document processing workflow:
- Loading and parsing documents
- Extracting metadata
- Splitting into chunks
- Embedding generation
- Storage to vector database
"""

from .document import RagDocument
from .pipeline import ingestion_pipeline

__all__ = ["RagDocument", "ingestion_pipeline"]
