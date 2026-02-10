import asyncio

from common.constants import VECTOR_SNOMED_KB
from repository.vector import vector_store
from rag.embedding import dense_embedder
from common import get_logger
from agent.entity import MedicalEntity

logger = get_logger(__name__)

SELECT_FIELDS = ["concept_id", "concept_name", "concept_code", "domain_id", "concept_class_id"]


async def search_medical_kb(medical_entity: MedicalEntity) -> None:
    """search ICD-10 and SNOMED standard concepts"""

    domain_mapping = {
        "diagnosis": "Condition",
        "procedure": "Procedure",
        "medication": "Drug",
        "symptom": "Observation",
    }

    embeddings = await asyncio.to_thread(
        dense_embedder.embed_queries_batch, [medical_entity.term_cn, medical_entity.term_en]
    )

    domain = domain_mapping.get(medical_entity.entity_type, "")

    icd_task = asyncio.to_thread(
        vector_store.search,
        selectFields=SELECT_FIELDS,
        dense_vectors=[embeddings[0]],
        limit=3,
        knowledgebaseIds=[VECTOR_SNOMED_KB],
        filters=f'vocabulary_id == "ICD10CN" and domain_id == "{domain}" and concept_class_id == "ICD10 code"',
    )
    snomed_task = asyncio.to_thread(
        vector_store.search,
        selectFields=SELECT_FIELDS,
        dense_vectors=[embeddings[1]],
        limit=3,
        knowledgebaseIds=[VECTOR_SNOMED_KB],
        filters=f'vocabulary_id == "SNOMED" and domain_id == "{domain}"',
    )

    icd_results, snomed_results = await asyncio.gather(icd_task, snomed_task)

    if icd_results and icd_results[0]:
        for icd in icd_results[0]:
            if medical_entity.icd10_concepts is None:
                medical_entity.icd10_concepts = {}
            concept_id = int(icd["concept_id"])
            medical_entity.icd10_concepts[concept_id] = {
                "concept_id": concept_id,
                "concept_name": icd["concept_name"],
                "concept_code": icd["concept_code"],
                "score": icd["score"],
            }

    if snomed_results and snomed_results[0]:
        for snomed in snomed_results[0]:
            if medical_entity.snomed_concepts is None:
                medical_entity.snomed_concepts = {}
            concept_id = int(snomed["concept_id"])
            medical_entity.snomed_concepts[concept_id] = {
                "concept_id": concept_id,
                "concept_name": snomed["concept_name"],
                "concept_code": snomed["concept_code"],
                "score": snomed["score"],
            }
