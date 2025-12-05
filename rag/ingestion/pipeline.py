import uuid

from llama_index.core.schema import Document
from rag.ingestion.parser import parse_content
from repository.vector.milvus_client import VectorStoreClient
import logging
from rag.ingestion.extractor.extractor import Extractor
from rag.ingestion.document import RagDocument
from rag.ingestion.parser.serializer_deserializer import serialize_documents
from rag.ingestion.splitter.markdown_splitter import RagMarkdownSplitter
from rag.core.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self):
        self.extractor = Extractor()
        self.splitter = RagMarkdownSplitter()
        self.embedder = EmbeddingService()
        self.vector_store = VectorStoreClient()

    def handle_document(self, filename: str, contents: bytes, content_type: str) -> RagDocument:
        try:
            documents = parse_content(contents, filename, content_type)
        except ValueError as e:
            raise ValueError(f"file parsing failed: {str(e)}")

        rag_document = self.build_from_parsed_documents(filename, contents, content_type, documents)

        # Step 2: Split document into chunks
        chunks = self.splitter.split_document(rag_document)
        logger.info(f"Document split into {len(chunks)} chunks")

        # Step 3: Embed chunks (generate vectors)
        chunks_with_vectors = self.embedder.embed_chunks(chunks)

        # Step 4: Prepare data for Milvus
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
                "kb_id": "default_kb",  # TODO: Support multi-KB
                "dense_vector": chunk.get("dense_vector", []),
                "text": chunk.get("text", ""),
                "business_data": rag_document.business_data or {},
                "upload_time": rag_document.upload_time,
            }
            chunks_to_insert.append(chunk_to_insert)

        # Step 5: Save to Milvus
        if chunks_to_insert:
            self.vector_store.insert(chunks_to_insert, "rag_fintech", "default_kb")
            logger.info(f"Saved {len(chunks_to_insert)} chunks to Milvus")

        return rag_document

    def build_from_parsed_documents(
        self, filename: str, contents: bytes, content_type: str, documents: list[Document]
    ) -> RagDocument:

        pages = serialize_documents(documents)
        # import json
        # import os

        # with open(f"api/db/{os.path.splitext(filename)[0]}.json", "r") as f:
        #     document = json.load(f)
        # pages = document["pages"]

        confidence, business_data = self.extractor.extract(pages)

        rag_document = RagDocument.from_extraction_result(
            document_id=str(uuid.uuid4()),
            parsed_documents=pages,
            confidence=confidence,
            business_data=business_data,
            filename=filename,
            file_size=len(contents),
            content_type=content_type,
        )

        logger.info(
            f"Built RagDocument for '{filename}': "
            f"{len(pages)} pages, "
            f"{len(business_data)} business_data fields on {rag_document.upload_time} UTC"
        )

        return rag_document
