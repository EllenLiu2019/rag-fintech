import json
import asyncio
from typing import Any
from collections import defaultdict

from langchain.tools import ToolRuntime, tool
from repository.graph import neo4j_client

from common import get_logger

logger = get_logger(__name__)


@tool
async def align_medical_concepts(runtime: ToolRuntime) -> str:
    """align ICD-10 and SNOMED concepts by direct mapping or path matching.

    This tool accepts AI-selected high-confidence concepts and performs alignment.
    Returns JSON string with aligned_concepts list, sorted by path_length (shortest first).
    """
    from agent.graph_state import AgentOutput, Step

    agent_output_dict: dict[str, AgentOutput] = runtime.state.agent_output_dict
    encode_agent_output: AgentOutput = agent_output_dict[Step.ENCODE_AGENT.value]

    icd10cn_concepts = encode_agent_output.step_output.get("icd10_concepts", [])
    snomed_concepts = encode_agent_output.step_output.get("snomed_concepts", [])

    if not icd10cn_concepts or not snomed_concepts:
        logger.warning(
            f"Missing concepts in encode_agent_output.result: icd10_concepts={bool(icd10cn_concepts)}, "
            f"snomed_concepts={bool(snomed_concepts)}"
        )
        return json.dumps({"aligned_concepts": []}, ensure_ascii=False)

    aligned_concepts: dict[str, dict[str, Any]] = defaultdict()

    # Build lookup dicts for concept_code (avoid N×M loop)
    icd_code_lookup = {c["concept_id"]: c["concept_code"] for c in icd10cn_concepts}
    snomed_code_lookup = {c["concept_id"]: c["concept_code"] for c in snomed_concepts}
    icd_ids = list(icd_code_lookup.keys())
    snomed_ids = list(snomed_code_lookup.keys())

    # Batch query 1: direct MAPS_TO (N×M → 1 query)
    direct_query = """
        MATCH (icd:Concept)-[:MAPS_TO]->(snomed:Concept)
        WHERE icd.id IN $icd_ids AND snomed.id IN $snomed_ids
        RETURN icd.id AS icd_id,
            icd.FSN AS icd_name,
            snomed.id AS snomed_id,
            snomed.FSN AS snomed_name
        ORDER BY icd.id
        """
    results = await asyncio.to_thread(
        neo4j_client.execute_query,
        direct_query,
        {"icd_ids": icd_ids, "snomed_ids": snomed_ids},
    )
    for result in results:
        key = f"{result['icd_id']}_{result['snomed_id']}"
        aligned_concepts[key] = {
            "path_length": 1,
            "icd_id": result["icd_id"],
            "icd_concept_code": icd_code_lookup[result["icd_id"]],
            "icd_name": result["icd_name"],
            "mapped_snomed_id": result["snomed_id"],
            "mapped_snomed_name": result["snomed_name"],
            "target_snomed_id": result["snomed_id"],
            "target_snomed_concept_code": snomed_code_lookup[result["snomed_id"]],
            "target_snomed_name": result["snomed_name"],
            "rel_types": ["MAPS_TO"],
        }

    # Batch query 2: path matching via ISA hierarchy (N×M → 1 query, only if no direct match)
    if not aligned_concepts:
        path_query = """
            MATCH path = (icd:Concept)-[:MAPS_TO]->(snomed1:Concept)-[:ISA*1..5]-(snomed2:Concept)
            WHERE icd.id IN $icd_ids
            AND snomed2.id IN $snomed_ids
            RETURN icd.id AS icd_id,
                icd.FSN AS icd_name,
                snomed1.id AS mapped_snomed_id,
                snomed1.FSN AS mapped_snomed_name,
                snomed2.id AS target_snomed_id,
                snomed2.FSN AS target_snomed_name,
                length(path) AS path_length,
                [rel in relationships(path) | type(rel)] AS rel_types
            ORDER BY icd.id, snomed2.id, path_length
            """
        results = await asyncio.to_thread(
            neo4j_client.execute_query,
            path_query,
            {"icd_ids": icd_ids, "snomed_ids": snomed_ids},
        )
        # Results are ordered by path_length; keep only the shortest path per (icd, snomed) pair
        for result in results:
            key = f"{result['icd_id']}_{result['target_snomed_id']}"
            if key in aligned_concepts:
                continue
            aligned_concepts[key] = {
                "path_length": int(result["path_length"]),
                "icd_id": result["icd_id"],
                "icd_concept_code": icd_code_lookup[result["icd_id"]],
                "icd_name": result["icd_name"],
                "mapped_snomed_id": result["mapped_snomed_id"],
                "mapped_snomed_name": result["mapped_snomed_name"],
                "target_snomed_id": result["target_snomed_id"],
                "target_snomed_concept_code": snomed_code_lookup[result["target_snomed_id"]],
                "target_snomed_name": result["target_snomed_name"],
                "rel_types": result["rel_types"],
            }

    if aligned_concepts:
        tool_call = {
            "name": "align_medical_concepts",
            "output": aligned_concepts,
        }
        agent_output_dict[Step.ALIGN_AGENT.value] = AgentOutput(
            name=Step.ALIGN_AGENT.value,
            tool_calls=[tool_call],
            agent_response={},
            step_output=None,
        )

    return json.dumps(
        {
            "aligned_concepts": (
                sorted(aligned_concepts.values(), key=lambda x: x["path_length"]) if aligned_concepts else []
            )
        },
        ensure_ascii=False,
        indent=2,
    )
