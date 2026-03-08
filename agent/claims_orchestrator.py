from typing import Dict, Any, List
import asyncio
import uuid
from datetime import datetime, timezone

from langsmith import traceable

from common import get_logger
from agent.medical_agents import graph, MedicalAgents
from agent.clause_matcher import ClauseMatcher
from agent.eligibility_reasoner import EligibilityReasoner
from agent.entity import ClaimRequest, ClaimDecision
from rag.persistence.persistent_service import PersistentService
from agent.graph_state import HumanDecision
from repository.rdb import rdb_client
from repository.rdb.models.models import ClaimEvaluations

logger = get_logger(__name__)


class ClaimsOrchestrator:
    """Claims Orchestrator - Coordinates multiple agents to evaluate claim requests

    Two-phase flow for human-in-the-loop:
        Phase 1: start_evaluation()  → runs medical agents to interrupt, returns pending review items
        Phase 2: complete_evaluation() → accepts human decisions, finishes clause matching & reasoning

    Each evaluation attempt is persisted to claim_evaluations for time-travel support.
    """

    def __init__(self):
        self.clause_matcher = ClauseMatcher()
        self.eligibility_reasoner = EligibilityReasoner()

    @traceable(run_type="chain", name="Start Claim Evaluation")
    async def start_evaluation(self, doc_id: str) -> Dict[str, Any]:
        """Phase 1: Run medical entity normalization until human review is needed.

        Returns:
            {
                "doc_id": str,
                "patient_id": str,
                "thread_ids": [str],     # thread IDs for each entity (needed to resume)
                "pending_reviews": [      # interrupt info for frontend display
                    {
                        "entity_index": int,
                        "entity_name": str,
                        "interrupts": [...]
                    }
                ]
            }
        """
        rdb_document = await PersistentService.aget_document(doc_id)
        request = ClaimRequest.from_dict(rdb_document.business_data)
        logger.info(f"Starting claim evaluation for patient: {request.patient_id}")

        # Generate unique thread_ids per evaluation attempt (supports re-evaluation)
        eval_id = uuid.uuid4().hex[:8]
        thread_ids = [f"{doc_id}_{eval_id}_entity_{i}" for i in range(len(request.medical_entities))]

        # Step 1: Medical Entity Normalization (run to interrupt)
        tasks = [graph.start(entity, thread_id) for entity, thread_id in zip(request.medical_entities, thread_ids)]
        results = await asyncio.gather(*tasks)

        # Extract interrupt info for frontend
        pending_reviews = []
        for i, (entity, result) in enumerate(zip(request.medical_entities, results)):
            interrupts = MedicalAgents.get_interrupts(result)
            if interrupts:
                pending_reviews.append(
                    {
                        "entity_index": i,
                        "entity_name": entity.term_cn,
                        "interrupts": interrupts,
                    }
                )

        # Capture subgraph configs from interrupted state (contains checkpoint_ns for time-travel)
        all_subgraph_configs = await graph.capture_subgraph_configs(thread_ids)

        # Persist evaluation records (status=reviewing, waiting for human decision)
        evaluations = [
            ClaimEvaluations(
                doc_id=doc_id,
                patient_id=request.patient_id,
                entity_index=i,
                entity_name=entity.term_cn,
                thread_id=thread_id,
                status="reviewing",
                subgraph_configs=all_subgraph_configs.get(thread_id),
            )
            for i, (entity, thread_id) in enumerate(zip(request.medical_entities, thread_ids))
        ]
        await asyncio.to_thread(rdb_client.save_all, evaluations)
        logger.info(f"Saved {len(evaluations)} evaluation records for doc_id={doc_id}")

        return {
            "doc_id": doc_id,
            "patient_id": request.patient_id,
            "thread_ids": thread_ids,
            "pending_reviews": pending_reviews,
        }

    @traceable(run_type="chain", name="Complete Claim Evaluation")
    async def complete_evaluation(
        self,
        doc_id: str,
        thread_ids: List[str],
        decisions: List[HumanDecision],
    ) -> ClaimDecision:
        """Phase 2: Resume with human decisions, then run clause matching & eligibility reasoning.

        Args:
            doc_id: Document ID
            thread_ids: Thread IDs from start_evaluation()
            decisions: Human-confirmed decisions for each entity
        """
        rdb_document = await PersistentService.aget_document(doc_id)
        request = ClaimRequest.from_dict(rdb_document.business_data)
        logger.info(f"Completing claim evaluation for patient: {request.patient_id}")

        # Update evaluation records: save human decisions and mark as approved
        now = datetime.now(timezone.utc)
        for thread_id, decision in zip(thread_ids, decisions):
            await asyncio.to_thread(
                self._update_evaluation,
                thread_id=thread_id,
                status="approved",
                human_decision=decision.to_dict(),
                updated_at=now,
            )

        # Resume medical agents with human decisions
        resume_tasks = [
            graph.resume(decision, graph._make_config(thread_id)) for decision, thread_id in zip(decisions, thread_ids)
        ]
        await asyncio.gather(*resume_tasks)

        # Step 2: Clause Matching
        evidence = await self.clause_matcher.match(
            entities=request.medical_entities, decisions=decisions, doc_id=request.policy_doc_id
        )

        # Step 3: Eligibility Reasoning
        reasoning_result = await self.eligibility_reasoner.reason(
            entities=request.medical_entities, decisions=decisions, evidence=evidence
        )

        # Step 4: Decision Generation
        claim_decision = self._generate_decision(evidence, reasoning_result)

        # Update evaluation records: mark as completed and persist decision result
        final_status = "completed" if claim_decision.status.value != "under_review" else "reviewing"
        now = datetime.now(timezone.utc)
        decision_dict = claim_decision.to_dict()
        for thread_id in thread_ids:
            await asyncio.to_thread(
                self._update_evaluation,
                thread_id=thread_id,
                status=final_status,
                decision_result=decision_dict,
                updated_at=now,
            )

        return claim_decision

    @staticmethod
    def _update_evaluation(thread_id: str, **kwargs) -> None:
        """Update a single evaluation record by thread_id."""
        record = rdb_client.select_by_kwargs(ClaimEvaluations, thread_id=thread_id)
        if not record:
            logger.warning(f"Evaluation record not found for thread_id={thread_id}")
            return
        record.update_from_dict(kwargs)
        rdb_client.save(record)

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
            matched_clauses=clauses_evidence,
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
    decision = asyncio.run(orchestrator.start_evaluation(doc_id))
    print(decision.to_dict())
