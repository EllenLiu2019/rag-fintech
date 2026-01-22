import asyncio
from typing import Dict, Any, List
import networkx as nx

from common import get_logger
from common.constants import VECTOR_GRAPH_KB
from rag.embedding import dense_embedder
from agent.tools.foc_retriever import foc_retriever
from rag.persistence import PersistentService
from rag.embedding import sparse_embedder
from repository.vector import vector_store
from repository.graph.neo4j_client import neo4j_client
from agent.entity import MedicalEntity
from rag.entity.clause_tree import ClauseForest


logger = get_logger(__name__)

EVIDENCE_TYPE_MAP = {
    "INCLUDE": "coverage",
    "NOT_INCLUDE": "exclusion",
}

TOP_K = 3


class ClauseMatcher:
    """
    clause matcher

    capabilities:
    1. FoC structured retrieval - based on clause directory to locate relevant sections
    2. Graph retrieval - based on entity relationships to find coverage/exclusion
    """

    async def match(self, entities: List[MedicalEntity], doc_id: str, claim_type: str) -> Dict[str, Any]:
        clause_forest = PersistentService.get_clause_forest(doc_id)

        tasks = [self._foc_retrieval(entities, clause_forest), self._graph_retrieval(entities, doc_id)]
        foc_result, graph_result = await asyncio.gather(*tasks)

        return await self._merge_results(foc_result, graph_result, clause_forest)

    async def _foc_retrieval(self, entities: List[MedicalEntity], clause_forest: ClauseForest) -> Dict[str, Any]:
        entity_names = set()
        icd10cn_names = set()
        snomed_names = set()
        for e in entities:
            entity_names.add(e.term_cn)
            icd10cn_names.add(e.aligned_concept.get("icd_name", ""))
            snomed_names.add(e.aligned_concept.get("target_snomed_name", ""))

        query = f"""
          请根据信息判断是否符合主险及附加险的赔付条件：
            诊断：{entity_names}
            ICD10CN：{icd10cn_names}
            SNOMED：{snomed_names}
            TNM分期：{', '.join([e.attributes.get("tnm_stage", "") for e in entities])}
        """

        return await asyncio.to_thread(foc_retriever.retrieve_candidate_chunks, query, clause_forest)

    async def _graph_retrieval(self, entities: List[MedicalEntity], doc_id: str) -> list[Dict[str, Any]]:
        graph_evidence = []

        for entity in entities:
            entity_matches = await self._search_graph_entities(entity, doc_id)
            if entity_matches:
                for match in entity_matches:
                    if not self._achieved_threshold(match["score"]):
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
                        evidence = self._extract_evidence_from_subgraph(
                            subgraph, match["entity_name"], match["root_id"]
                        )
                        graph_evidence.append(evidence)

        return graph_evidence

    def _achieved_threshold(self, score: float, k: int = 60) -> bool:
        threshold = 2 / (k + TOP_K - 1)
        return score >= threshold

    async def _search_graph_entities(self, entity: MedicalEntity, doc_id: str) -> List[Dict[str, Any]]:

        tnm_stage = entity.attributes.get("tnm_stage", "")
        diagnosis = entity.aligned_concept.get("icd_name", "")
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

    def _extract_evidence_from_subgraph(self, subgraph, entity_name: str, root_id: int) -> Dict[str, Any]:
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

    async def _merge_results(
        self,
        foc_result: Dict[str, Any],
        graph_result: list[Dict[str, Any]],
        clause_forest: ClauseForest,
    ) -> Dict[str, Any]:
        coverage_evidence = []
        exclusion_evidence = []
        for result in graph_result:
            coverages = result.get("coverage", [])
            if coverages:
                for coverage in coverages:
                    coverage_str = ""
                    for cvg in coverage:
                        coverage_str += f"{cvg['source']} --{cvg['relation']}--> "
                    coverage_str += f"{coverage[-1]['target']}"
                    coverage_evidence.append(coverage_str)

            exclusions = result.get("exclusion", [])
            if exclusions:
                for exclusion in exclusions:
                    exclusion_str = ""
                    for excl in exclusion:
                        exclusion_str += f"{excl['source']} --{excl['relation']}--> "
                    exclusion_str += f"{exclusion[-1]['target']}"
                    exclusion_evidence.append(exclusion_str)

        clause_ids = set()
        clause_ids.update(foc_result.get("clause_ids", []))
        for result in graph_result:
            clause_ids.update(result.get("clause_ids", []))

        clause_evidence = await self._get_clause_evidence(clause_ids, clause_forest)

        return {
            "coverage_evidence": coverage_evidence,
            "exclusion_evidence": exclusion_evidence,
            "clause_ids": list(clause_ids),
            "clauses_evidence": clause_evidence,
        }

    async def _get_clause_evidence(self, clause_ids: set[int], clause_forest: ClauseForest) -> str:
        clause_evidence = ""
        clause_paths = set[int]()
        for clause_id in clause_ids:
            node = clause_forest.root.reverse_find_node(clause_id)
            for clause_path in node.build_clause_path().split("."):
                clause_paths.add(int(clause_path))

        for clause_path in sorted(clause_paths):
            node = clause_forest.root.reverse_find_node(clause_path)
            if node:
                header = "#" * (node.level + 2)
                clause_evidence += f"{header} {node.title} \n"
                clause_evidence += f"  - {node.content}\n"

        return clause_evidence
