import uuid
from typing import Any

from llama_index.core.schema import Document

from common import config
from rag.ingestion.parser import parse_content
from rag.ingestion.extractor.extractor import Extractor
from rag.ingestion.document import RagDocument
from rag.ingestion.parser.serializer_deserializer import serialize_documents
from rag.ingestion.splitter.markdown_splitter import RagMarkdownSplitter
from rag.core.embedding_service import EmbeddingService
from rag.ingestion.doc_service import DocumentService

import logging

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, chat_model: dict[str, Any]):
        self.document_service = DocumentService()
        self.extractor = Extractor(chat_model)
        self.splitter = RagMarkdownSplitter()
        self.vector_store = config.VECTOR_STORE

    def handle_document(self, filename: str, contents: bytes, content_type: str):

        # Step 1: Save file to rdb
        rdb_document = self.document_service.upload_file(filename, contents, content_type)

        # Step 2: Parse document
        try:
            documents = parse_content(contents, filename, content_type)
        except ValueError as e:
            raise ValueError(f"file parsing failed: {str(e)}")

        rag_document = self.build_from_parsed_documents(filename, contents, content_type, documents)

        # Step 3: Split document into chunks
        chunks = self.splitter.split_document(rag_document)
        logger.info(f"Document split into {len(chunks)} chunks")

        # Step 4: Embed chunks (generate vectors)
        llm = self.document_service.get_embedding_model(rdb_document.kb_name)
        embedder = EmbeddingService(provider=llm.llm_provider, model_name=llm.model_name)
        chunks_with_vectors = embedder.embed_chunks(chunks, rag_document)

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
            self.vector_store.insert(chunks_to_insert, "rag_fintech", rdb_document.kb_name)
            logger.info(f"Saved {len(chunks_to_insert)} chunks to Milvus")

        # Step 7: Update doc info in rdb
        self.document_service.update_file_info(filename, rag_document, rdb_document)

    def build_from_parsed_documents(
        self, filename: str, contents: bytes, content_type: str, documents: list[Document]
    ) -> RagDocument:

        pages = serialize_documents(documents)
        # import json
        # import os

        # with open(f"repository/s3/parsed_files/{os.path.splitext(filename)[0]}.json", "r") as f:
        #     document = json.load(f)
        # pages = document["pages"]

        confidence, business_data, tokens = self.extractor.extract(pages)

        rag_document = RagDocument.from_extraction_result(
            document_id=str(uuid.uuid4()),
            parsed_documents=pages,
            confidence=confidence,
            business_data=business_data,
            token_num=tokens,
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
