import logging
import json
from typing import Any, Optional
from copy import deepcopy

from rag.ingestion.extractor.rule_extractor import RuleExtractor
from rag.ingestion.extractor.llm_extractor import LLMExtractor
from rag.ingestion.utils.confidence_calculator import ConfidenceCalculator
from rag.ingestion.extractor.metadata_creator import MetadataCreator

logger = logging.getLogger(__name__)


class Extractor:
    """
    Multi-strategy extraction pipeline: rule + pattern + LLM
    - Rule-based extraction: Strong format fields
    - LLM-based extraction: Complex fields
    """

    def __init__(self, chat_model: dict[str, Any]) -> None:
        self.llm_extractor: LLMExtractor = LLMExtractor(chat_model)
        self.rule_extractor: RuleExtractor = RuleExtractor()
        self.confidence_calculator: ConfidenceCalculator = ConfidenceCalculator()
        self.metadata_creator = MetadataCreator()

    def extract(self, documents: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Args:
            documents: list of documents, format: [dict[str, Any], ...]

        Returns:
            tuple[dict[str, Any], dict[str, Any]]: confidence_result, metadata
        """
        if not isinstance(documents, list) or not all(isinstance(doc, dict) for doc in documents):
            raise ValueError("documents must be a list of dict objects")

        try:
            logger.info("Starting rule-based extraction")

            raw_documents = deepcopy(documents)
            self.rule_extractor.extract(raw_documents)

            logger.info("Calculating confidence scores")
            confidence_result = self.confidence_calculator.calculate(
                self.rule_extractor.extracted_result,
                self.rule_extractor.schema,
                self.rule_extractor.extraction_signals,
            )
            metadata = self.metadata_creator.create(self.rule_extractor.extracted_result)

            tokens = 0
            if confidence_result.get("overall_confidence", 0.0) < 0.8:
                logger.warning(f"Low confidence detected: {confidence_result.get('overall_confidence')}. ")
                llm_results = self._fallback_to_llm_extraction(documents, metadata)
                llm_content = json.loads(llm_results["content"].replace("```json", "").replace("```", ""))
                tokens = llm_results["tokens"]
                metadata_llm = self.metadata_creator.create(llm_content)
                metadata.update(metadata_llm)

            return confidence_result, metadata, tokens

        except Exception as e:
            logger.error(f"Extraction pipeline failed: {e}", exc_info=True)
            raise

    def _fallback_to_llm_extraction(
        self, documents: list[dict[str, Any]], metadata: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        try:
            hints = deepcopy(metadata)

            content = "\n".join([doc.get("text", "") for doc in documents])

            llm_results = self.llm_extractor.extract(content, hints=hints)
            logger.info("LLM extraction completed")
            return llm_results

        except Exception as e:
            logger.error(f"LLM extraction failed: {e}", exc_info=True)
            return None

    def get_extracted_result(self) -> dict[str, Any]:
        return self.rule_extractor.extracted_result

    def reset(self) -> None:
        self.rule_extractor.extracted_result = {}
        self.rule_extractor.extraction_signals = {}
        self.rule_extractor.grids = []
        logger.info("Extractor reset")
