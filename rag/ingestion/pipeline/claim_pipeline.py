import asyncio

from common import get_logger
from .base_pipeline import BasePipeline
from rag.ingestion.parser import ParseResult
from rag.entity import RagDocument, DocumentType
from rag.retrieval.pre_optimizer import GlossaryInjector
from rag.marshaller import serialize_batch
from rag.persistence import PersistentService

logger = get_logger(__name__)


class ClaimPipeline(BasePipeline):
    """
    claim document processing pipeline

    1. parse PDF
    2. extract metadata (medical entities)
    3. persist to file system
    """

    def __init__(self, doc_type: DocumentType = DocumentType.CLAIM):
        super().__init__(doc_type)
        self.glossary_injector = GlossaryInjector()

    def validate_input(self, filename: str, content_type: str):
        super().validate_input(filename, content_type)

        if content_type != "application/pdf":
            raise ValueError(f"Claim materials must be PDF, got: {content_type}")

    async def extract_information(
        self,
        filename: str,
        contents: bytes,
        parse_result: ParseResult,
        **kwargs,
    ) -> RagDocument:
        """
        extract medical entities
        """
        document_id = kwargs.get("document_id")

        logger.info(f"Extracting medical entities for document: {document_id}")

        pages = serialize_batch(parse_result.documents)

        rag_document = RagDocument.from_extraction_result(
            document_id=document_id,
            parsed_documents=pages,
            business_data={},
            confidence={},
            token_num=0,
            filename=filename,
            file_size=len(contents),
            content_type=parse_result.content_type,
            job_id=parse_result.job_id,
            doc_type=self.doc_type,
        )

        return rag_document

    async def post_process(self, rag_document: RagDocument) -> None:
        """
        post-process: normalize medical entities with SNOMED
        """
        logger.info("Normalizing medical entities with SNOMED")

        # normalize medical entities with SNOMED
        # snomed_entities = await self.glossary_injector.ner(text_content)

    async def persist(self, rag_document: RagDocument, **kwargs) -> None:
        """
        persist to rdb
        """
        doc_id = rag_document.document_id
        rdb_document_id = kwargs.get("rdb_document_id")

        await asyncio.to_thread(PersistentService.update_document, rag_document, rdb_document_id)
        logger.info(f"RDB document updated for document: {doc_id}")
