import asyncio
from typing import Dict, Any, List
import networkx as nx

from common import get_logger
from common.constants import VECTOR_GRAPH_KB
from rag.embedding import dense_embedder, sparse_embedder
from repository.vector import vector_store
from repository.graph.neo4j_client import neo4j_client
from agent.entity import MedicalEntity

logger = get_logger(__name__)

EVIDENCE_TYPE_MAP = {
    "INCLUDE": "coverage",
    "NOT_INCLUDE": "exclusion",
}

TOP_K = 3


def _achieved_threshold(score: float, k: int = 60) -> bool:
    threshold = 2 / (k + TOP_K - 1)
    return score >= threshold


async def _search_graph_entities(entity: MedicalEntity, doc_id: str) -> List[Dict[str, Any]]:
    agent_reasoning = entity.agent_reasoning
    aligned_concept = agent_reasoning.get("aligned_concept", {})
    diagnosis = aligned_concept.get("icd_name", "")
    tnm_stage = agent_reasoning.get("tnm_stage", "")
    if tnm_stage:
        diagnosis += f"（{tnm_stage}）"

    search_results = []
    dense_task = asyncio.to_thread(dense_embedder.embed_query, diagnosis)
    sparse_task = asyncio.to_thread(sparse_embedder.embed_queries, [diagnosis])
    dense_vector, sparse_vectors = await asyncio.gather(dense_task, sparse_task)

    results = await asyncio.to_thread(
        vector_store.hybrid_search,
        selectFields=["entity_name", "entity_type", "description", "root_id"],
        dense_vectors=[dense_vector],
        sparse_vectors=sparse_vectors,
        limit=TOP_K,
        knowledgebaseIds=[VECTOR_GRAPH_KB],
        filters={"doc_id": doc_id, "graph_type": "entity"},
    )

    if results and results[0]:
        for result in results[0]:
            r = {
                "entity_name": result.get("entity_name", ""),
                "entity_type": result.get("entity_type", ""),
                "description": result.get("description", ""),
                "root_id": result.get("root_id", ""),
                "score": result.get("score", 0),
            }
            search_results.append(r)

    return search_results


def _extract_evidence_from_subgraph(subgraph, entity_name: str, root_id: int) -> Dict[str, Any]:
    evidence = {"coverage": [], "exclusion": []}
    clause_ids = set[int]()
    start_id = None
    target_id = None
    for node_id, node_data in subgraph.nodes(data=True):
        if node_data.get("entity_name") == "保险责任" and node_data.get("root_id") == root_id:
            start_id = node_id
        if node_data.get("entity_name") == entity_name and node_data.get("root_id") == root_id:
            target_id = node_id
        if start_id and target_id:
            break

    all_paths = list(nx.all_simple_paths(subgraph, start_id, target_id, cutoff=5))
    for path in all_paths:
        evidence_item = []
        is_coverage = True
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            source_name = subgraph.nodes[source].get("entity_name")
            target_name = subgraph.nodes[target].get("entity_name")
            edge_data = subgraph.get_edge_data(source, target)
            rel_type = edge_data.get("rel_type")

            for node in [source, target]:
                node_clause_ids = subgraph.nodes[node].get("clause_ids", [])
                for clause_id in node_clause_ids:
                    clause_ids.add(int(clause_id))

            if rel_type in ["NOT_INCLUDE"]:
                is_coverage = False
            evidence_item.append(
                {
                    "source": source_name,
                    "target": target_name,
                    "relation": rel_type,
                    "description": edge_data.get("description", ""),
                    "root_id": edge_data.get("root_id"),
                }
            )
        if is_coverage:
            evidence["coverage"].append(evidence_item)
        else:
            evidence["exclusion"].append(evidence_item)

    return {
        "coverage": evidence["coverage"],
        "exclusion": evidence["exclusion"],
        "clause_ids": clause_ids,
    }


async def graph_retrieval(entities: List[MedicalEntity], doc_id: str) -> list[Dict[str, Any]]:
    """
    Perform graph-based retrieval to find coverage/exclusion evidence.

    Args:
        entities: List of medical entities to search for
        doc_id: Document ID to search within

    Returns:
        List of evidence dictionaries containing coverage, exclusion, and clause_ids
    """
    graph_evidence = []

    for entity in entities:
        entity_matches = await _search_graph_entities(entity, doc_id)
        if entity_matches:
            for match in entity_matches:
                if not _achieved_threshold(match["score"]):
                    continue
                subgraph = await asyncio.to_thread(
                    neo4j_client.get_relationship_subgraph,
                    "保险责任",
                    match["entity_name"],
                    doc_id,
                    match["root_id"],
                    EVIDENCE_TYPE_MAP.keys(),
                )
                if subgraph.number_of_nodes() > 0:
                    evidence = _extract_evidence_from_subgraph(subgraph, match["entity_name"], match["root_id"])
                    graph_evidence.append(evidence)

    return graph_evidence
