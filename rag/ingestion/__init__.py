"""
Document ingestion pipeline.

This module handles the complete document processing workflow:
- Loading and parsing documents
- Extracting metadata
- Splitting into chunks
- Embedding generation
- Storage to vector database
"""

from .pipeline import ingestion_pipeline
from .tasks import enqueue_task, get_task

__all__ = ["ingestion_pipeline", "enqueue_task", "get_task"]
