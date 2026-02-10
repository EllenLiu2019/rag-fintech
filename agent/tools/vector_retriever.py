from typing import List, Dict, Any

from common import get_logger
from common.constants import VECTOR_DEFAULT_KB
from rag.retrieval.retriever import retriever
from agent.entity import MedicalEntity
from agent.graph_state import HumanDecision

logger = get_logger(__name__)


async def vector_retrieval(
    entities: List[MedicalEntity],
    decisions: List[HumanDecision],
    doc_id: str,
) -> Dict[str, Any]:
    """
    Perform vector retrieval based on medical entities.

    Args:
        entities: List of medical entities to search for
        doc_id: Document ID to filter results

    Returns:
        Dict with clause_paths and chunks
    """
    # Build query from entities
    entity_names = [e.term_cn for e in entities]
    icd10cn_names = []
    snomed_names = []
    tnm_stages = []
    for decision in decisions:
        tnm_stages.append(decision.tnm_stage)
        icd10cn_names.append(decision.icd_concept_name)

    query_parts = []
    if entity_names:
        query_parts.append(f"诊断：{', '.join(entity_names)}")
    if icd10cn_names:
        query_parts.append(f"ICD10CN：{', '.join(icd10cn_names)}")
    if snomed_names:
        query_parts.append(f"SNOMED：{', '.join(snomed_names)}")
    if tnm_stages:
        query_parts.append(f"TNM分期：{', '.join(tnm_stages)}")

    query = " ".join(query_parts)

    if not query:
        logger.warning("Empty query generated from entities")
        return []

    filters = {"doc_id": doc_id} if doc_id else None
    vector_results = await retriever._retrieve_vector_results(
        optimized_queries=[query],
        kb_id=VECTOR_DEFAULT_KB,
        top_k=5,
        filters=filters,
        mode="hybrid",
    )

    clause_paths = []
    chunks = []
    if vector_results and vector_results[0]:
        for result in vector_results[0]:
            clause_path = result.get("clause_path", "")
            if clause_path:
                clause_paths.append(clause_path)
            else:
                chunks.append(result.get("text", ""))
    logger.info(f"Found {len(clause_paths)} clause paths from vector retrieval.")

    return {"clause_paths": clause_paths, "chunks": chunks}
