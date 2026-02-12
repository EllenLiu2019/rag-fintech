from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
import asyncio
from rq import get_current_job

from common import get_logger
from rag.ingestion.parser import aparse_content, ParseResult
from repository.s3 import s3_client
from rag.entity import RagDocument, DocumentType
from common.exceptions import ParsingError, IngestionError
from common.error_codes import ErrorCodes
from rag.ingestion.tasks import enqueue_task
from graphrag import index as graphrag_index
from rag.entity.clause_tree import ClauseForest

logger = get_logger(__name__)


class IngestionResult:

    def __init__(self, doc_id: str, doc_type: DocumentType, status: str, metadata: Dict[str, Any] = None):
        self.doc_id = doc_id
        self.doc_type = doc_type
        self.status = status
        self.metadata = metadata or {}


class BasePipeline(ABC):

    def __init__(self, doc_type: DocumentType):
        self.doc_type = doc_type

    async def process(
        self,
        filename: str,
        content_type: str,
        callback: Callable[[str, int, str], None],
        **kwargs,
    ) -> None:
        logger.info(f"Starting {self.doc_type.value} ingestion for: {filename}")

        try:
            job_id = self._get_job_id()

            if job_id and callback:
                callback(job_id, 1, "Validating input")

            self.validate_input(filename, content_type)

            if job_id and callback:
                callback(job_id, 2, "Loading file from S3")

            contents = await self.load_original_file(filename, **kwargs)

            if job_id and callback:
                callback(job_id, 3, "Parsing document")

            parse_result = await self.parse_document(contents, filename, content_type)

            if job_id and callback:
                callback(job_id, 4, "Extracting information")

            rag_document = await self.extract_information(filename, contents, parse_result, **kwargs)

            if job_id and callback:
                callback(job_id, 5, "Post-processing")

            await self.post_process(rag_document)

            if job_id and callback:
                callback(job_id, 6, "Persisting")

            await self.persist(rag_document, **kwargs)

            if job_id and callback:
                callback(job_id, 7, "Completed")

            if kwargs.get("graph_enabled", False):
                if self.doc_type == DocumentType.POLICY and rag_document.clause_forest:
                    graph_job_id = enqueue_task(
                        build_graph,
                        rag_document.clause_forest,
                        job_timeout=1800,
                        document_id=rag_document.document_id,
                    )
                    logger.info(f"Enqueued graph build job: {graph_job_id}")

            logger.info(f"Completed {self.doc_type.value} ingestion: {rag_document.document_id}")

        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
            raise

    def _get_job_id(self) -> str:
        try:
            current_job = get_current_job()
            job_id = current_job.id if current_job else None
        except Exception as e:
            logger.warning(f"Failed to get current job: {e}")
            job_id = None

        return job_id

    def validate_input(self, filename: str, content_type: str) -> None:
        if not filename:
            raise ValueError("Filename is required")

        if not content_type:
            raise ValueError("Content type is required")

    async def load_original_file(self, filename: str, **kwargs) -> bytes:
        return await asyncio.to_thread(
            s3_client.load_original_file, filename, kwargs.get("document_id"), self.doc_type.value
        )

    async def parse_document(self, contents: bytes, filename: str, content_type: str):
        logger.info(f"Parsing document: {filename}")
        try:
            parse_result = await aparse_content(contents, filename, content_type)
        except ValueError as e:
            raise ParsingError(
                message=f"Failed to parse document: {filename}",
                code=ErrorCodes.S_INGESTION_002,
                details={"filename": filename, "content_type": content_type, "reason": str(e)},
            )
        return parse_result

    @abstractmethod
    async def extract_information(
        self,
        filename: str,
        contents: bytes,
        parse_result: ParseResult,
        **kwargs,
    ) -> RagDocument:
        pass

    @abstractmethod
    async def post_process(self, rag_document: RagDocument) -> None:
        """
        - 保单：分块、向量化
        - 理赔：实体标准化
        """
        pass

    @abstractmethod
    async def persist(self, rag_document: RagDocument, **kwargs) -> None:
        """
        - 保单：向量库 + RDB + Graph
        - 理赔：RDB + 文件系统
        """
        pass


def build_graph(clause_forest: ClauseForest, **kwargs):
    document_id = kwargs.get("document_id")
    logger.info(f"Starting graph build for document: {document_id}")

    try:
        asyncio.run(graphrag_index.index(document_id, clause_forest))

        logger.info(f"Completed graph build for document: {document_id}")
    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        raise IngestionError(message=f"Failed to build graph: {e}", code=ErrorCodes.S_INGESTION_010)
