import asyncio

from .base_pipeline import BasePipeline
from rag.ingestion.extractor import extractor
from rag.ingestion.parser import ParseResult
from rag.ingestion.indexing.markdown_splitter import RagMarkdownSplitter
from rag.embedding import dense_embedder, sparse_embedder
from rag.persistence import PersistentService
from repository.vector import vector_store
from rag.marshaller import serialize_batch
from rag.entity import RagDocument, DocumentType
from common import get_logger
from common.exceptions import ExtractionError, VectorStoreError, ChunkingError, EmbeddingError
from common.error_codes import ErrorCodes
from common.constants import VECTOR_DEFAULT_KB

logger = get_logger(__name__)


class PolicyPipeline(BasePipeline):
    """
    policy document processing pipeline

    1. parse PDF
    2. extract metadata (business_data, confidence, clause_forest)
    3. chunk (Markdown Splitter)
    4. embed (Dense + Sparse)
    5. persist to vector store
    6. persist to rdb
    7. build knowledge graph (GraphRAG)
    """

    def __init__(self, doc_type: DocumentType = DocumentType.POLICY):
        super().__init__(doc_type)

    def validate_input(self, filename: str, content_type: str) -> None:
        super().validate_input(filename, content_type)

        if content_type not in ["application/pdf", "text/plain"]:
            raise ValueError(f"Unsupported content type for policy: {content_type}")

    async def extract_information(
        self,
        filename: str,
        contents: bytes,
        parse_result: ParseResult,
        **kwargs,
    ) -> RagDocument:
        document_id = kwargs.get("document_id")

        logger.info(f"Extracting policy key information for document: {document_id}")

        pages = serialize_batch(parse_result.documents)

        try:
            confidence, business_data, tokens, clause_forest = await asyncio.to_thread(
                extractor.extract, documents=pages, source_file=filename
            )
        except Exception as e:
            raise ExtractionError(
                message=f"Failed to extract metadata from document: {filename}",
                code=ErrorCodes.S_INGESTION_003,
                details={"filename": filename, "error": str(e)},
            )

        rag_document = RagDocument.from_extraction_result(
            document_id=document_id,
            parsed_documents=pages,
            confidence=confidence,
            business_data=business_data,
            token_num=tokens,
            filename=filename,
            file_size=len(contents),
            content_type=parse_result.content_type,
            job_id=parse_result.job_id,
            clause_forest=clause_forest,
            doc_type=self.doc_type,
        )

        logger.info(
            f"Built RagDocument for '{filename}': "
            f"{len(pages)} pages, "
            f"{len(business_data)} business_data fields on {rag_document.upload_time} UTC"
        )

        return rag_document

    async def post_process(self, rag_document: RagDocument) -> None:
        logger.info("Post-processing policy: chunking and embedding")

        try:
            splitter = RagMarkdownSplitter()
            await asyncio.to_thread(splitter.split_document, doc=rag_document)
            logger.info(f"Document split into {len(rag_document.chunks)} chunks")
        except Exception as e:
            raise ChunkingError(
                message=f"Failed to split document into chunks: {rag_document.filename}",
                code=ErrorCodes.S_INGESTION_004,
                details={"filename": rag_document.filename, "error": str(e), "error_type": type(e).__name__},
            )

        try:
            await self._embed_chunks(rag_document)
            logger.info(f"Chunks embedded for document: {rag_document.document_id}")
        except Exception as e:
            raise EmbeddingError(
                message=f"Failed to embed chunks for document: {rag_document.filename}",
                code=ErrorCodes.L_EMBEDDING_001,
                details={"filename": rag_document.filename, "error": str(e), "error_type": type(e).__name__},
            )

    async def _embed_chunks(self, rag_document: RagDocument) -> None:
        chunks = rag_document.chunks
        await asyncio.gather(
            asyncio.to_thread(dense_embedder.embed_chunks, chunks, rag_document),
            asyncio.to_thread(sparse_embedder.embed_chunks, chunks),
        )

    async def persist(self, rag_document: RagDocument, **kwargs) -> None:
        doc_id = rag_document.document_id
        rdb_document_id = kwargs.get("rdb_document_id")

        await self._persist_vector_store(rag_document)
        logger.info(f"Vector store persisted for document: {doc_id}")

        await PersistentService.aupdate_document(rag_document, rdb_document_id)
        logger.info(f"RDB document updated for document: {doc_id}")

    async def _persist_vector_store(self, rag_document: RagDocument) -> None:
        # Prepare data for Milvus
        chunks_to_insert = []
        for chunk in rag_document.chunks:
            chunk_metadata = chunk.get("metadata", {})

            chunk_to_insert = {
                "id": chunk["chunk_id"],
                "doc_id": rag_document.document_id,
                "file_name": rag_document.filename,
                "page_number": chunk_metadata.get("page_number", 0),
                "prev_chunk": chunk.get("prev_chunk", None),
                "next_chunk": chunk.get("next_chunk", None),
                "kb_id": VECTOR_DEFAULT_KB,
                "dense_vector": chunk.get("dense_vector", []),
                "sparse_vector": chunk.get("sparse_vector", {}),
                "text": chunk.get("text", ""),
                "business_data": rag_document.business_data or {},
                "clause_id": chunk.get("clause_id"),
                "clause_path": chunk.get("clause_path"),
                "upload_time": rag_document.upload_time,
            }
            chunks_to_insert.append(chunk_to_insert)

        try:
            await asyncio.to_thread(vector_store.insert, chunks_to_insert, VECTOR_DEFAULT_KB)
            logger.info(f"Saved {len(chunks_to_insert)} chunks to Milvus")
        except Exception as e:
            raise VectorStoreError(
                message=f"Failed to save chunks to vector store: {rag_document.filename}",
                code=ErrorCodes.R_VECTOR_002,
                details={"filename": rag_document.filename, "chunk_count": len(chunks_to_insert), "error": str(e)},
            )
