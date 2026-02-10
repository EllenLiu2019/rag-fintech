from fastapi import UploadFile
from typing import List, Optional

from common import get_logger
from common.exceptions import (
    DatabaseError,
    FileStorageError,
    ModelNotFoundError,
    DocumentNotFoundError,
)
from common.error_codes import ErrorCodes
from common.constants import VECTOR_DEFAULT_KB
from repository.rdb.models.models import Document as RdbDocument, KnowledgeBase, LLM, ClaimEvaluations
from repository.rdb import rdb_client
from rag.entity import RagDocument, ClauseForest
from repository.s3 import s3_client
from repository.cache import cached
from rag.marshaller import deserialize, serialize

logger = get_logger(__name__)


class PersistentService:
    @staticmethod
    async def upload_file(file: UploadFile, doc_type: str) -> RdbDocument:
        logger.info(f"Uploading file: {file.filename}")
        contents = await file.read()

        try:
            storage_file = s3_client.save_original_file(file.filename, contents, doc_type)
        except Exception as e:
            raise FileStorageError(
                message=f"Failed to save original file: {file.filename}",
                code=ErrorCodes.R_FILE_001,
                details={"filename": file.filename, "doc_type": doc_type, "error": str(e)},
            ) from e

        document = RdbDocument(
            document_id=storage_file.doc_id,
            file_name=storage_file.filename,
            doc_type=storage_file.doc_type,
            file_location=storage_file.file_path,
            content_type=file.content_type,
            file_size=len(contents),
            doc_status="uploaded",
            kb_name=VECTOR_DEFAULT_KB,
        )

        try:
            rdb_document = rdb_client.save(document)
        except Exception as e:
            raise DatabaseError(
                message=f"Failed to save document to database: {file.filename}",
                code=ErrorCodes.R_DB_002,
                details={"filename": file.filename, "doc_type": doc_type, "error": str(e)},
            ) from e

        logger.info(
            f"File {file.filename} saved to RDB: id={rdb_document.id} with original file location: {storage_file.file_path}"
        )

        return rdb_document

    @staticmethod
    def update_document(rag_document: RagDocument, rdb_id: int):
        file_name = rag_document.filename
        logger.info(f"Updating file {file_name} in RDB: id={rdb_id}")

        parsed_file = rag_document.to_parsed_file()
        try:
            storage_file = s3_client.save_parsed_file(file_name, parsed_file)
        except Exception as e:
            raise FileStorageError(
                message=f"Failed to save parsed file info: {file_name}",
                code=ErrorCodes.R_FILE_001,
                details={"filename": file_name, "doc_type": parsed_file.get("doc_type"), "error": str(e)},
            ) from e

        # Update document attributes
        rdb_document = rdb_client.select_by_id(RdbDocument, rdb_id)
        if rdb_document is None:
            raise DocumentNotFoundError(
                message=f"Document not found: {rag_document.document_id} (rdb_id: {rdb_id})",
                code=ErrorCodes.R_DB_003,
                details={"rdb_id": rdb_id},
            )

        rdb_document.file_name = file_name
        rdb_document.doc_status = "completed"
        rdb_document.doc_location = storage_file.file_path
        rdb_document.content_type = rag_document.content_type
        rdb_document.page_count = len(rag_document.pages)
        rdb_document.upload_time = rag_document.upload_time
        rdb_document.business_data = rag_document.business_data
        rdb_document.confidence = rag_document.confidence
        rdb_document.token_num += rag_document.token_num if rag_document.token_num is not None else 0
        rdb_document.chunk_num += rag_document.chunk_num if rag_document.chunk_num is not None else 0
        rdb_document.clause_forest = serialize(rag_document.clause_forest) if rag_document.clause_forest else None

        # Save the updated document (merge will update existing record)
        try:
            updated_doc = rdb_client.save(rdb_document)

            logger.info(
                f"File {file_name} updated in RDB: id={updated_doc.id} with parsed file location: {storage_file.file_path}"
            )
        except Exception as e:
            raise DatabaseError(
                message=f"Failed to update document in database: {file_name}",
                code=ErrorCodes.R_DB_002,
                details={"filename": file_name, "document_id": rdb_document.id, "error": str(e)},
            ) from e

    @staticmethod
    def get_embedding_model(kb_name: str) -> str:
        try:
            kb_ids = rdb_client.execute_query(KnowledgeBase, kb_name)
            if not kb_ids:
                raise ModelNotFoundError(
                    message=f"Knowledge base not found: {kb_name}",
                    code=ErrorCodes.L_MODEL_001,
                    details={"kb_name": kb_name},
                )
            kb_id = kb_ids[0]
            llm_model = rdb_client.select_by_id(LLM, kb_id.embed_llm_id)
            if llm_model is None:
                raise ModelNotFoundError(
                    message=f"LLM model not found for knowledge base: {kb_name}",
                    code=ErrorCodes.L_MODEL_001,
                    details={"kb_name": kb_name, "embed_llm_id": kb_id.embed_llm_id},
                )
            return llm_model
        except ModelNotFoundError:
            raise
        except Exception as e:
            raise DatabaseError(
                message=f"Failed to query embedding model from database: {kb_name}",
                code=ErrorCodes.R_DB_002,
                details={"kb_name": kb_name, "error": str(e)},
            ) from e

    @staticmethod
    def list_documents(doc_type: str = "all") -> List[RdbDocument]:
        """List all documents, optionally filtered by doc_type."""
        try:
            if doc_type == "all":
                return rdb_client.select(RdbDocument)
            return rdb_client.select_all_by_kwargs(RdbDocument, doc_type=doc_type)
        except Exception as e:
            raise DatabaseError(
                message=f"Failed to list documents with doc_type={doc_type}",
                code=ErrorCodes.R_DB_002,
                details={"doc_type": doc_type, "error": str(e)},
            ) from e

    @staticmethod
    def list_evaluations(doc_id: str) -> List[ClaimEvaluations]:
        """List all evaluation records for a given doc_id."""
        try:
            return rdb_client.select_all_by_kwargs(ClaimEvaluations, doc_id=doc_id)
        except Exception as e:
            raise DatabaseError(
                message=f"Failed to list evaluations for doc_id={doc_id}",
                code=ErrorCodes.R_DB_002,
                details={"doc_id": doc_id, "error": str(e)},
            ) from e

    @staticmethod
    @cached(prefix="get_foc", ttl=3600)
    def get_clause_forest(doc_id: str) -> Optional[ClauseForest]:
        rdb_document = rdb_client.select_by_kwargs(RdbDocument, document_id=doc_id)
        if rdb_document is None:
            return None
        return deserialize(rdb_document.clause_forest, target_type=ClauseForest)

    @staticmethod
    @cached(prefix="get_document", ttl=3600)
    def get_document(doc_id: str) -> RdbDocument:
        rdb_document = rdb_client.select_by_kwargs(RdbDocument, document_id=doc_id)
        if rdb_document is None:
            raise DocumentNotFoundError(
                message=f"Document not found: {doc_id}",
                code=ErrorCodes.R_DB_003,
                details={"doc_id": doc_id},
            )
        return rdb_document
