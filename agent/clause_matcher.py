import asyncio
from typing import Dict, Any, List

from common import get_logger
from agent.tools.foc_retriever import foc_retrieval
from agent.tools.graph_retriever import graph_retrieval
from agent.tools.vector_retriever import vector_retrieval
from rag.persistence import PersistentService
from agent.entity import MedicalEntity
from rag.entity.clause_tree import ClauseForest
from agent.graph_state import HumanDecision


logger = get_logger(__name__)


class ClauseMatcher:
    """
    clause matcher

    capabilities:
    1. FoC structured retrieval - based on clause directory to locate relevant sections
    2. Graph retrieval - based on entity relationships to find coverage/exclusion
    3. Vector retrieval - based on entity names to find policy information.
    """

    async def match(self, entities: List[MedicalEntity], decisions: List[HumanDecision], doc_id: str) -> Dict[str, Any]:
        logger.info(f"Matching clauses for document: {doc_id}")

        clause_forest = await PersistentService.aget_clause_forest(doc_id)
        logger.info(f"Clause forest for document: {doc_id} retrieved, size: {len(clause_forest.trees)}")

        tasks = [
            foc_retrieval(entities, decisions, clause_forest),
            graph_retrieval(decisions, doc_id),
            vector_retrieval(entities, decisions, doc_id),
        ]
        foc_result, graph_result, vector_result = await asyncio.gather(*tasks)

        return await self._merge_results(
            foc_result,
            graph_result,
            clause_forest,
            vector_result,
        )

    async def _merge_results(
        self,
        foc_result: Dict[str, Any],
        graph_result: list[Dict[str, Any]],
        clause_forest: ClauseForest,
        vector_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.info("Merging results")

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
        for clause_path in vector_result.get("clause_paths", []):
            for id in clause_path.split("."):
                clause_ids.add(int(id))

        chunks = vector_result.get("chunks", [])
        clause_evidence = await self._get_clause_evidence(clause_ids, chunks, clause_forest)

        return {
            "coverage_evidence": coverage_evidence,
            "exclusion_evidence": exclusion_evidence,
            "clause_ids": list(clause_ids),
            "clauses_evidence": clause_evidence,
        }

    async def _get_clause_evidence(
        self,
        clause_ids: set[int],
        chunks: List[str],
        clause_forest: ClauseForest,
    ) -> str:
        clause_evidence = ""
        for chunk in chunks:
            clause_evidence += f"{chunk}\n"

        clause_paths = set[int]()
        for clause_id in clause_ids:
            node = clause_forest.root.reverse_find_node(clause_id)
            if node is None:
                logger.warning(f"Clause node not found for id={clause_id}, skipping")
                continue
            for clause_path in node.build_clause_path().split("."):
                clause_paths.add(int(clause_path))

        for clause_path in sorted(clause_paths):
            node = clause_forest.root.reverse_find_node(clause_path)
            if node:
                header = "#" * (node.level + 2)
                clause_evidence += f"{header} {node.title} \n"
                clause_evidence += f"  - {node.content}\n"

        return clause_evidence


if __name__ == "__main__":
    import asyncio
    from agent.entity import MedicalEntity
    import json

    medical_entity = MedicalEntity(
        patient_age=56,
        term_cn="甲状腺乳头状癌",
        term_en="papillary thyroid carcinoma",
        entity_type="diagnosis",
        attributes={
            "tumor_max_diameter_cm": 1.2,
            "is_lymph_metastasis": False,
        },
        description="甲状腺乳头状癌，肿瘤位置: 右叶下极，肿瘤大小: 1.2 cm × 1.0 cm，被膜侵犯: (-)，脉管侵犯: (-)，神经侵犯: (-)，中央区淋巴结未见癌转移 (0/6).",
    )

    clause_matcher = ClauseMatcher()
    human_decisions = [HumanDecision(icd_concept_code="C73.x00", icd_concept_name="甲状腺恶性肿瘤", tnm_stage="I期")]
    results = asyncio.run(clause_matcher.match([medical_entity], human_decisions, "policy_0119223547_a02169"))
    print(json.dumps(results, ensure_ascii=False, indent=2))
