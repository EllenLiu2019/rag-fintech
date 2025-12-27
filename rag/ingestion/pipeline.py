import uuid
from typing import Any
from fastapi import UploadFile

from rag.ingestion.parser import parse_content
from rag.ingestion.extractor.extractor import Extractor
from rag.entity import RagDocument
from rag.ingestion.parser.parser import ParseResult
from rag.ingestion.parser.serializer_deserializer import serialize_documents
from rag.ingestion.splitter.markdown_splitter import RagMarkdownSplitter
from rag.core import embedder
from repository.vector import vector_store
from common import get_logger, get_model_registry
from common.exceptions import (
    ParsingError,
    ExtractionError,
    ChunkingError,
    EmbeddingError,
    VectorStoreError,
)
from common.error_codes import ErrorCodes
from repository.s3 import s3_client
from rag.ingestion.tasks import enqueue_task
from rag.ingestion.storage_service import StorageService

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(self, model: dict[str, Any]):
        self.storage_service = StorageService()
        self.extractor = Extractor(model)

    async def __call__(self, file: UploadFile):
        rdb_document = await self.storage_service.upload_file(file)

        # Enqueue task - RQ will generate job_id
        # Use module-level function to avoid pickle issues with instance methods
        job_id = enqueue_task(
            handle_document,  # Module-level function, not instance method
            file.filename,
            file.content_type,
            rdb_document.kb_name,
            rdb_document.id,
        )

        logger.info(f"Enqueued document ingestion job: {job_id}")
        return job_id

    def build_from_parsed_documents(
        self, filename: str, contents: bytes, content_type: str, parse_result: ParseResult
    ) -> RagDocument:

        pages = serialize_documents(parse_result.documents)
        # import json
        # import os

        # with open(f"repository/s3/parsed_files/{os.path.splitext(filename)[0]}.json", "r") as f:
        #     document = json.load(f)
        # pages = document["pages"]

        try:
            confidence, business_data, tokens, clause_forest = self.extractor.extract(
                documents=pages, source_file=filename
            )
        except Exception as e:
            raise ExtractionError(
                message=f"Failed to extract metadata from document: {filename}",
                code=ErrorCodes.S_INGESTION_003,
                details={"filename": filename, "error": str(e)},
            )

        rag_document = RagDocument.from_extraction_result(
            document_id=str(uuid.uuid4()),
            parsed_documents=pages,
            confidence=confidence,
            business_data=business_data,
            token_num=tokens,
            filename=filename,
            file_size=len(contents),
            content_type=content_type,
            job_id=parse_result.job_id,
            clause_forest=clause_forest,
        )

        logger.info(
            f"Built RagDocument for '{filename}': "
            f"{len(pages)} pages, "
            f"{len(business_data)} business_data fields on {rag_document.upload_time} UTC"
        )

        return rag_document


def handle_document(
    filename: str,
    content_type: str,
    kb_name: str,
    rdb_document_id: int,
):
    """
    Module-level function to process document ingestion pipeline.
    This function is called by RQ worker and re-initializes the pipeline instance.

    Args:
        filename: File name
        content_type: Content type
        kb_name: Knowledge base name
        rdb_document_id: RDB document ID
    """
    # Import here to avoid circular imports and pickle issues
    from rag.ingestion.tasks import update_progress
    from rq import get_current_job

    # Get the singleton pipeline instance (will be recreated in worker if needed)
    pipeline = ingestion_pipeline

    try:
        current_job = get_current_job()
        job_id = current_job.id if current_job else None
        logger.info(f"Processing document {filename} with job ID: {job_id}")
    except Exception as e:
        logger.warning(f"Failed to get current job: {e}")
        job_id = None

    if job_id:
        update_progress(job_id, 1, "Loading file from S3")

    # Parse document
    contents = s3_client.load_original_file(filename)
    if job_id:
        update_progress(job_id, 2, "Parsing document")

    try:
        parse_result = parse_content(contents, filename, content_type)
    except ValueError as e:
        raise ParsingError(
            message=f"Failed to parse document: {filename}",
            code=ErrorCodes.S_INGESTION_002,
            details={"filename": filename, "content_type": content_type, "reason": str(e)},
        )

    if job_id:
        update_progress(job_id, 3, "Extracting metadata")

    rag_document = pipeline.build_from_parsed_documents(filename, contents, content_type, parse_result)

    if job_id:
        update_progress(job_id, 4, "Splitting document into chunks")

    try:
        chunks = RagMarkdownSplitter().split_document(doc=rag_document)
        logger.info(f"Document split into {len(chunks)} chunks")
    except Exception as e:
        raise ChunkingError(
            message=f"Failed to split document into chunks: {filename}",
            code=ErrorCodes.S_INGESTION_004,
            details={"filename": filename, "error": str(e), "error_type": type(e).__name__},
        )

    # Embed chunks (generate vectors)
    if job_id:
        update_progress(job_id, 5, f"Embedding {len(chunks)} chunks")

    try:
        chunks_with_vectors = embedder.embed_chunks(chunks, rag_document)
    except Exception as e:
        if isinstance(e, EmbeddingError):
            raise
        raise EmbeddingError(
            message=f"Failed to embed document chunks: {filename}",
            code=ErrorCodes.L_EMBEDDING_001,
            details={"filename": filename, "chunk_count": len(chunks), "error": str(e)},
        )

    # Prepare data for Milvus
    chunks_to_insert = []
    for chunk in chunks_with_vectors:
        chunk_metadata = chunk.get("metadata", {})

        chunk_to_insert = {
            "id": chunk["chunk_id"],
            "doc_id": rag_document.document_id,
            "file_name": rag_document.filename,
            "page_number": chunk_metadata.get("page_number", 0),
            "prev_chunk": chunk.get("prev_chunk", None),
            "next_chunk": chunk.get("next_chunk", None),
            "kb_id": kb_name,
            "dense_vector": chunk.get("dense_vector", []),
            "text": chunk.get("text", ""),
            "business_data": rag_document.business_data or {},
            "clause_id": chunk.get("clause_id"),
            "clause_path": chunk.get("clause_path"),
            "clause_title": chunk.get("clause_title"),
            "upload_time": rag_document.upload_time,
        }
        chunks_to_insert.append(chunk_to_insert)

    # Save to Milvus
    if chunks_to_insert:
        if job_id:
            update_progress(job_id, 6, f"Saving {len(chunks_to_insert)} chunks to Milvus")

        try:
            vector_store.insert(chunks_to_insert, kb_name)
            logger.info(f"Saved {len(chunks_to_insert)} chunks to Milvus")
        except Exception as e:
            raise VectorStoreError(
                message=f"Failed to save chunks to vector store: {filename}",
                code=ErrorCodes.R_VECTOR_002,
                details={"filename": filename, "chunk_count": len(chunks_to_insert), "error": str(e)},
            )

    # Update doc info in rdb
    if job_id:
        update_progress(job_id, 7, "Updating document info in RDB")

    try:
        pipeline.storage_service.update_file_info(filename, rag_document, rdb_document_id)
        if job_id:
            update_progress(job_id, 8, "Document processing completed!")
    except Exception as e:
        # Log but don't fail the entire ingestion if RDB update fails
        logger.warning(f"Failed to update document info in RDB: {e}", exc_info=True)


def _create_ingestion_pipeline() -> IngestionPipeline:
    """
    Create IngestionPipeline instance at module load time.
    """
    registry = get_model_registry()
    model_config = registry.get_chat_model("query_lite")
    ingestion_pipeline = IngestionPipeline(model=model_config.to_dict())

    logger.info("Initialized IngestionPipeline singleton")
    return ingestion_pipeline


ingestion_pipeline = _create_ingestion_pipeline()
