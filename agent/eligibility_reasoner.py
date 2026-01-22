import json
from typing import Dict, Any, List
import asyncio

from common import get_logger
from common.prompt_manager import get_prompt_manager
from rag.llm.chat_model import chat_model
from common import get_model_registry
from agent.entity import MedicalEntity

logger = get_logger(__name__)


class EligibilityReasoner:
    """
    core capabilities:
    1. based on graph/clause/vector evidence for decision making
    2. generate explainable reasoning chain
    """

    def __init__(self):
        registry = get_model_registry()
        model_config = registry.get_chat_model("qa_reasoner")
        model = model_config.to_dict()

        self.llm = chat_model[model["provider"]](
            model_name=model["model_name"],
            base_url=model["base_url"],
        )
        self.prompt_manager = get_prompt_manager()

    async def reason(
        self,
        entities: List[MedicalEntity],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:

        context = self._build_reasoning_context(entities, evidence)

        prompt = self.prompt_manager.get("claim_reasoning", **context)

        try:
            reasoning, content, tokens = await asyncio.to_thread(
                self.llm.generate,
                messages=[{"role": "system", "content": prompt}],
            )

            result = json.loads(content.replace("```json", "").replace("```", ""))

            return {
                "decision": result.get("decision", ""),
                "explanation": result.get("explanation", ""),
                "recommendations": result.get("recommendations", []),
                "reasoning": reasoning,
                "tokens": tokens,
            }

        except Exception as e:
            logger.error(f"Final reasoning failed: {e}")
            return {
                "decision": "NEED_MANUAL_REVIEW",
                "explanation": f"自动推理失败，需要人工审核: {str(e)}",
                "recommendations": ["请人工审核理赔申请"],
            }

    def _build_reasoning_context(
        self,
        entities: List[MedicalEntity],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        patient_conditions = [
            {
                "diagnosis": e.term_cn,
                "icd10cn_concepts": e.aligned_concept.get("icd_name", ""),
                "snomed_concepts": e.aligned_concept.get("target_snomed_name", ""),
                "attributes": e.attributes,
            }
            for e in entities
        ]
        coverage_evidence = evidence.get("coverage_evidence", [])
        exclusion_evidence = evidence.get("exclusion_evidence", [])
        clauses_evidence = evidence.get("clauses_evidence", "")
        return {
            "patient_conditions": json.dumps(patient_conditions, ensure_ascii=False, indent=2),
            "coverage_evidence": json.dumps(coverage_evidence, ensure_ascii=False, indent=2),
            "exclusion_evidence": json.dumps(exclusion_evidence, ensure_ascii=False, indent=2),
            "clauses_evidence": json.dumps(clauses_evidence, ensure_ascii=False, indent=2),
        }
