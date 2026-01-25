from langchain.tools import tool
import asyncio
from typing import Any
from pydantic import BaseModel, Field
import json

from repository.graph import neo4j_client
from common import get_logger

logger = get_logger(__name__)


class MedicalConcepts(BaseModel):
    icd10cn_concepts: list[dict[str, Any]] = Field(
        ..., description="ICD10CN concepts（AI-selected high-confidence concepts）"
    )
    snomed_concepts: list[dict[str, Any]] = Field(
        ..., description="SNOMED concepts（AI-selected high-confidence concepts）"
    )


@tool(args_schema=MedicalConcepts)
async def align_medical_concepts(icd10cn_concepts: list[dict[str, Any]], snomed_concepts: list[dict[str, Any]]) -> str:
    """align ICD-10 and SNOMED concepts by direct mapping or path matching.

    This tool accepts AI-selected high-confidence concepts and performs alignment.
    Returns JSON string with aligned_concepts list, sorted by path_length (shortest first).
    """

    aligned_concepts = []

    direct_query = """
        MATCH (icd:Concept)-[:MAPS_TO]->(snomed:Concept)
        WHERE icd.id = $icd10cn_id AND snomed.id = $snomed_id
        RETURN icd.id AS icd_id,
            icd.FSN AS icd_name,
            snomed.id AS snomed_id,
            snomed.FSN AS snomed_name
        ORDER BY icd.id
        """
    for icd_concept in icd10cn_concepts:
        icd10cn_id = icd_concept.get("concept_id")
        icd_concept_code = icd_concept.get("concept_code")
        for snomed_concept in snomed_concepts:
            snomed_id = snomed_concept.get("concept_id")
            snomed_concept_code = snomed_concept.get("concept_code")

            results = await asyncio.to_thread(
                neo4j_client.execute_query,
                direct_query,
                {"icd10cn_id": icd10cn_id, "snomed_id": snomed_id},
            )
            for result in results:
                aligned_concepts.append(
                    {
                        "path_length": 1,
                        "icd_id": result["icd_id"],
                        "icd_concept_code": icd_concept_code,
                        "icd_name": result["icd_name"],
                        "mapped_snomed_id": result["snomed_id"],
                        "mapped_snomed_name": result["snomed_name"],
                        "target_snomed_id": result["snomed_id"],
                        "target_snomed_concept_code": snomed_concept_code,
                        "target_snomed_name": result["snomed_name"],
                        "rel_types": ["MAPS_TO"],
                    }
                )
    if not aligned_concepts:
        query = """
            MATCH path = (icd:Concept)-[:MAPS_TO]->(snomed1:Concept)-[:ISA*1..5]-(snomed2:Concept)
            WHERE icd.id = $icd10cn_id
            AND snomed2.id = $snomed_id
            RETURN icd.id AS icd_id,
                icd.FSN AS icd_name,
                snomed1.id AS mapped_snomed_id,
                snomed1.FSN AS mapped_snomed_name,
                snomed2.id AS target_snomed_id,
                snomed2.FSN AS target_snomed_name,
                length(path) AS path_length,
                [rel in relationships(path) | type(rel)] AS rel_types
            ORDER BY path_length
            LIMIT 1
            """

        for icd_concept in icd10cn_concepts:
            icd10cn_id = icd_concept.get("concept_id")
            icd_concept_code = icd_concept.get("concept_code")

            for snomed_concept in snomed_concepts:
                snomed_id = snomed_concept.get("concept_id")
                snomed_concept_code = snomed_concept.get("concept_code")

                results = await asyncio.to_thread(
                    neo4j_client.execute_query,
                    query,
                    {"icd10cn_id": icd10cn_id, "snomed_id": snomed_id},
                )
                for result in results:
                    aligned_concepts.append(
                        {
                            "path_length": int(result["path_length"]),
                            "icd_id": result["icd_id"],
                            "icd_concept_code": icd_concept_code,
                            "icd_name": result["icd_name"],
                            "mapped_snomed_id": result["mapped_snomed_id"],
                            "mapped_snomed_name": result["mapped_snomed_name"],
                            "target_snomed_id": result["target_snomed_id"],
                            "target_snomed_concept_code": snomed_concept_code,
                            "target_snomed_name": result["target_snomed_name"],
                            "rel_types": result["rel_types"],
                        }
                    )

    aligned_concepts = sorted(aligned_concepts, key=lambda x: x["path_length"])

    return json.dumps({"aligned_concepts": aligned_concepts}, ensure_ascii=False)
