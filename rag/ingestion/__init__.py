"""
Document ingestion pipeline.

This module handles the complete document processing workflow:
- Loading and parsing documents
- Extracting metadata
- Splitting into chunks
- Embedding generation
- Storage to vector database
"""
from .pipeline import IngestionPipeline
from .document import RagDocument

__all__ = ["IngestionPipeline", "RagDocument"]

