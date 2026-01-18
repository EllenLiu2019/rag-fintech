import asyncio
import json
from typing import Any

from common import get_logger
from rag.ingestion.pipeline.base_pipeline import BasePipeline
from rag.ingestion.parser import ParseResult
from rag.entity import RagDocument, DocumentType
from rag.marshaller import serialize_batch
from rag.persistence import PersistentService
from rag.llm.chat_model import chat_model
from common.prompt_manager import get_prompt_manager
from common.model_registry import get_model_registry

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
        registry = get_model_registry()
        model_config = registry.get_chat_model("qa_reasoner")
        model = model_config.to_dict()
        self.llm = chat_model[model["provider"]](
            model_name=model["model_name"],
            base_url=model["base_url"],
        )
        self.prompt_manager = get_prompt_manager()

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

        extract_result = await self._medical_extract(pages)

        rag_document = RagDocument.from_extraction_result(
            document_id=document_id,
            parsed_documents=pages,
            business_data=extract_result["content"],
            confidence={"overall_confidence": extract_result["reasoning"]},
            token_num=extract_result["tokens"],
            filename=filename,
            file_size=len(contents),
            content_type=parse_result.content_type,
            job_id=parse_result.job_id,
            doc_type=self.doc_type,
        )

        return rag_document

    async def _medical_extract(self, pages: list[dict[str, Any]]) -> dict[str, Any]:
        """
        extract medical entities
        """
        text = "\n".join([page["text"] for page in pages])
        prompt = self.prompt_manager.get("claim_extraction", text=text)
        reasoning, content, tokens = await asyncio.to_thread(
            self.llm.generate,
            messages=[{"role": "system", "content": prompt}],
        )
        content = json.loads(content.replace("```json", "").replace("```", ""))
        return {
            "content": content,
            "tokens": tokens,
            "reasoning": reasoning,
        }

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
