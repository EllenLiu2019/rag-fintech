import asyncio
import json
from typing import Any
from datetime import datetime, timezone

from common import get_logger
from rag.ingestion.pipeline.base_pipeline import BasePipeline
from rag.ingestion.parser import ParseResult
from rag.entity import RagDocument, DocumentType
from rag.marshaller import serialize_batch
from rag.persistence import PersistentService
from rag.llm.chat_model import chat_model
from common.prompt_manager import get_prompt_manager
from common.model_registry import get_model_registry
from agent.entity import MedicalEntity, ClaimRequest

logger = get_logger(__name__)

MANDATORY_FIELDS = [
    "name",
    "gender",
    "age",
    "hospital",
    "report_date",
    "primary_diagnosis",
    "diagnosis_en",
    "pathology_details",
    "diagnosis_description",
]


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
        missing_data = self._validate_content(content)
        if missing_data:
            logger.info(f"Extracting missing data: {missing_data}")
            missing_prompt = self.prompt_manager.get(
                "missing_data", content=content, missing_data=json.dumps(missing_data)
            )
            history = [{"role": "system", "content": prompt}, {"role": "assistant", "content": content}]
            history.append({"role": "user", "content": missing_prompt})
            reasoning, content, tokens = await asyncio.to_thread(
                self.llm.generate,
                messages=history,
            )
            content = json.loads(content.replace("```json", "").replace("```", ""))
        return {
            "content": content,
            "tokens": tokens,
            "reasoning": reasoning,
        }

    def _validate_content(self, content: dict[str, Any]) -> dict[str, Any]:
        """
        validate content
        """
        missing_fields = [field for field in MANDATORY_FIELDS if not content.get(field)]
        return missing_fields

    async def post_process(self, rag_document: RagDocument) -> None:
        """
        post-process: normalize medical entities with SNOMED
        """
        logger.info("Normalizing medical entities with SNOMED")

        # normalize medical entities with SNOMED
        claim_data = rag_document.business_data
        medical_entity = MedicalEntity(
            entity_type="diagnosis",
            patient_age=claim_data["age"],
            term_cn=claim_data["primary_diagnosis"],
            term_en=claim_data["diagnosis_en"],
            attributes=claim_data.get("pathology_details", {}),
            description=claim_data["diagnosis_description"],
        )
        claim_request = ClaimRequest(
            patient_id=claim_data["name"],
            patient_age=claim_data["age"],
            policy_doc_id="policy_0119223547_a02169",
            medical_entities=[medical_entity],
            claim_type="medical",
            claim_date=datetime.now(tz=timezone.utc).isoformat(),
        )
        rag_document.business_data = claim_request.to_dict()
        # claim_decision = await self.claims_orchestrator.evaluate_claim(claim_request)

    async def persist(self, rag_document: RagDocument, **kwargs) -> None:
        """
        persist to rdb
        """
        doc_id = rag_document.document_id
        rdb_document_id = kwargs.get("rdb_document_id")

        await asyncio.to_thread(PersistentService.update_document, rag_document, rdb_document_id)
        logger.info(f"RDB document updated for document: {doc_id}")
