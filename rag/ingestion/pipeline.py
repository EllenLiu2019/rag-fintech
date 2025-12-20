import uuid
from typing import Any

from llama_index.core.schema import Document

from rag.ingestion.parser import parse_content
from rag.ingestion.extractor.extractor import Extractor
from rag.ingestion.document import RagDocument
from rag.ingestion.parser.parser import ParseResult
from rag.ingestion.parser.serializer_deserializer import serialize_documents
from rag.ingestion.splitter.markdown_splitter import RagMarkdownSplitter
from rag.core.embedding_service import EmbeddingService
from rag.core.doc_service import DocumentService
from repository.vector.milvus_client import VectorStoreClient

from common import get_logger, get_model_registry
from repository.rdb.models.models import LLM
from common.exceptions import (
    ParsingError,
    ExtractionError,
    ChunkingError,
    EmbeddingError,
    VectorStoreError,
)
from common.error_codes import ErrorCodes

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(self, model: dict[str, Any]):
        self.document_service = DocumentService()
        self.vector_store = VectorStoreClient()
        self.extractor = Extractor(model)
        self.splitter = RagMarkdownSplitter()

    def handle_document(self, filename: str, contents: bytes, content_type: str):

        # Step 1: Save file to rdb
        rdb_document = self.document_service.upload_file(filename, contents, content_type)

        # Step 2: Parse document
        try:
            parse_result = parse_content(contents, filename, content_type)
        except ValueError as e:
            raise ParsingError(
                message=f"Failed to parse document: {filename}",
                code=ErrorCodes.S_INGESTION_002,
                details={"filename": filename, "content_type": content_type, "reason": str(e)},
            )

        rag_document = self.build_from_parsed_documents(filename, contents, content_type, parse_result)

        # Step 3: Split document into chunks
        try:
            chunks = self.splitter.split_document(rag_document)
            logger.info(f"Document split into {len(chunks)} chunks")
        except Exception as e:
            raise ChunkingError(
                message=f"Failed to split document into chunks: {filename}",
                code=ErrorCodes.S_INGESTION_004,
                details={"filename": filename, "error": str(e)},
            )

        # Step 4: Embed chunks (generate vectors)
        try:
            llm: LLM = self.document_service.get_embedding_model(rdb_document.kb_name)
            embedder = EmbeddingService(model=llm.to_dict())
            chunks_with_vectors = embedder.embed_chunks(chunks, rag_document)
        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise
            raise EmbeddingError(
                message=f"Failed to embed document chunks: {filename}",
                code=ErrorCodes.L_EMBEDDING_001,
                details={"filename": filename, "chunk_count": len(chunks), "error": str(e)},
            )

        # Step 5: Prepare data for Milvus
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
                "kb_id": rdb_document.kb_name,
                "dense_vector": chunk.get("dense_vector", []),
                "text": chunk.get("text", ""),
                "business_data": rag_document.business_data or {},
                "upload_time": rag_document.upload_time,
            }
            chunks_to_insert.append(chunk_to_insert)

        # Step 6: Save to Milvus
        if chunks_to_insert:
            try:
                self.vector_store.insert(chunks_to_insert, "rag_fintech", rdb_document.kb_name)
                logger.info(f"Saved {len(chunks_to_insert)} chunks to Milvus")
            except Exception as e:
                raise VectorStoreError(
                    message=f"Failed to save chunks to vector store: {filename}",
                    code=ErrorCodes.R_VECTOR_002,
                    details={"filename": filename, "chunk_count": len(chunks_to_insert), "error": str(e)},
                )

        # Step 7: Update doc info in rdb
        try:
            self.document_service.update_file_info(filename, rag_document, rdb_document)
        except Exception as e:
            # Log but don't fail the entire ingestion if RDB update fails
            logger.warning(f"Failed to update document info in RDB: {e}", exc_info=True)

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
            confidence, business_data, tokens = self.extractor.extract(pages)
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
        )

        logger.info(
            f"Built RagDocument for '{filename}': "
            f"{len(pages)} pages, "
            f"{len(business_data)} business_data fields on {rag_document.upload_time} UTC"
        )

        return rag_document


def _create_ingestion_pipeline() -> IngestionPipeline:
    """
    Create IngestionPipeline instance at module load time.
    """
    registry = get_model_registry()
    model_config = registry.get_chat_model("qa_lite")
    ingestion_pipeline = IngestionPipeline(model=model_config.to_dict())

    logger.info("Initialized IngestionPipeline singleton")
    return ingestion_pipeline


ingestion_pipeline = _create_ingestion_pipeline()
