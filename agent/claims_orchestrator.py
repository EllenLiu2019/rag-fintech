from typing import Dict, Any, List
import asyncio

from langsmith import traceable

from common import get_logger
from agent.medical_agent import MedicalAgent
from agent.clause_matcher import ClauseMatcher
from agent.eligibility_reasoner import EligibilityReasoner
from agent.entity import ClaimRequest, ClaimDecision, MedicalEntity
from rag.persistence.persistent_service import PersistentService


logger = get_logger(__name__)


class ClaimsOrchestrator:
    """Claims Orchestrator - Coordinates multiple agents to evaluate claim requests"""

    def __init__(self):
        self.medical_agent = MedicalAgent()
        self.clause_matcher = ClauseMatcher()
        self.eligibility_reasoner = EligibilityReasoner()

    @traceable(run_type="chain", name="Evaluate Claim Pipeline")
    async def evaluate_claim(self, doc_id: str) -> ClaimDecision:
        """
        Evaluate claim request
        Pipeline:
        1. Medical entity normalization → SNOMED graph concept reasoning
        2. Clause matching → find related coverage and exclusion clauses
        3. Eligibility reasoning → based on graph reasoning whether satisfies conditions
        4. Decision generation → output explainable claim decision
        """
        rdb_document = PersistentService.get_document(doc_id)
        request = ClaimRequest.from_dict(rdb_document.business_data)
        logger.info(f"Evaluating claim for patient: {request.patient_id}")

        # Step 1: Medical Entity Normalization
        await self._normalize_medical_entities(request.medical_entities)

        # Step 2: Clause Matching
        evidence = await self._match_clauses(request.medical_entities, request.policy_doc_id)

        # Step 3: Eligibility Reasoning
        reasoning_result = await self._reason_eligibility(request.medical_entities, evidence)

        # Step 4: Decision Generation
        decision = self._generate_decision(evidence, reasoning_result)

        return decision

    @traceable(run_type="chain", name="Normalize Entities")
    async def _normalize_medical_entities(self, entities: List[MedicalEntity]) -> List[MedicalEntity]:
        tasks = [self.medical_agent.run(entity) for entity in entities]
        return await asyncio.gather(*tasks)

    @traceable(run_type="tool", name="Match Clauses")
    async def _match_clauses(self, entities: List[MedicalEntity], policy_doc_id: str) -> Dict[str, Any]:
        return await self.clause_matcher.match(entities=entities, doc_id=policy_doc_id)

    @traceable(run_type="llm", name="Reason Eligibility")
    async def _reason_eligibility(self, entities: List[MedicalEntity], evidence: Dict[str, Any]) -> Dict[str, Any]:
        return await self.eligibility_reasoner.reason(entities=entities, evidence=evidence)

    def _generate_decision(self, evidence: Dict[str, Any], reasoning: Dict[str, Any]) -> ClaimDecision:
        from agent.entity import ClaimStatus, ClaimDecision

        status = reasoning.get("status", ClaimStatus.UNDER_REVIEW)
        if isinstance(status, str):
            status = ClaimStatus(status)

        eligible_items = evidence.get("coverage_evidence", [])
        excluded_items = evidence.get("exclusion_evidence", [])
        clauses_evidence = evidence.get("clauses_evidence", "")

        decision = ClaimDecision(
            status=status,
            eligible_items=eligible_items,
            excluded_items=excluded_items,
            matched_clauses=clauses_evidence,  # Markdown string format
            explanation=reasoning.get("explanation", ""),
            recommendations=reasoning.get("recommendations", []),
            reasoning=reasoning.get("reasoning", ""),
            tokens=reasoning.get("tokens", 0),
        )

        logger.info(f"Generated claim decision: status={status.value}, tokens={reasoning.get('tokens', 0)}")
        return decision


if __name__ == "__main__":

    doc_id = "claim_0120164504_f58784"
    orchestrator = ClaimsOrchestrator()
    decision = asyncio.run(orchestrator.evaluate_claim(doc_id))
    print(decision.to_dict())
