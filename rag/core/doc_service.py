from typing import List

from common import get_logger
from repository.rdb.models.models import Document as RdbDocument, KnowledgeBase, LLM
from repository.rdb.postgresql_client import PostgreSQLClient
from rag.ingestion.document import RagDocument
from repository.s3 import s3_client

logger = get_logger(__name__)


class DocumentService:
    def __init__(self):
        self.rdb_client = PostgreSQLClient()

    def upload_file(self, filename: str, contents: bytes, content_type: str):
        logger.info(f"uploading file {filename}")

        file_path = s3_client.save_original_file(filename, contents)
        if content_type == "application/pdf":
            kb_name = "default_kb"
        else:
            kb_name = "unknown"

        # kb_ids = self.rdb_client.execute_query(KnowledgeBase, kb_name)
        # if len(kb_ids) == 0:
        #     raise ValueError(f"Knowledge base {kb_name} not found")

        document = RdbDocument(
            file_name=filename,
            file_location=file_path,
            content_type=content_type,
            file_size=len(contents),
            doc_status="uploaded",
            kb_name=kb_name,
        )
        rdb_document = self.rdb_client.save(document)

        logger.info(f"file {filename} saved to rdb: id={rdb_document.id} with original file location: {file_path}")

        return rdb_document

    def update_file_info(self, filename: str, rag_document: RagDocument, rdb_document: RdbDocument):
        logger.info(f"updating file {filename} in rdb: id={rdb_document.id}")

        parsed_file = rag_document.to_parsed_file()
        parsed_file_path = s3_client.save_file_info(filename, parsed_file)

        # Update document attributes
        rdb_document.document_id = rag_document.document_id
        rdb_document.file_name = rag_document.filename
        rdb_document.doc_status = "completed"
        rdb_document.doc_location = parsed_file_path
        rdb_document.content_type = rag_document.content_type
        rdb_document.page_count = len(rag_document.pages)
        rdb_document.upload_time = rag_document.upload_time
        rdb_document.business_data = rag_document.business_data
        rdb_document.confidence = rag_document.confidence
        rdb_document.token_num += rag_document.token_num
        rdb_document.chunk_num += rag_document.chunk_num
        # Save the updated document (merge will update existing record)
        updated_doc = self.rdb_client.save(rdb_document)

        logger.info(
            f"file {filename} updated in rdb: id={updated_doc.id} with parsed file location: {parsed_file_path}"
        )

    def get_embedding_model(self, kb_name: str) -> str:
        kb_ids = self.rdb_client.execute_query(KnowledgeBase, kb_name)
        kb_id = kb_ids[0]
        llm_model = self.rdb_client.select_by_id(LLM, kb_id.embed_llm_id)
        if llm_model is None:
            raise ValueError(f"LLM model not found for knowledge base {kb_id.kb_name}")
        return llm_model

    def download_file(self, filename: str) -> bytes:
        return s3_client.load_original_file(filename)

    def list_files(self) -> List[str]:
        return s3_client.list_stored_files()
