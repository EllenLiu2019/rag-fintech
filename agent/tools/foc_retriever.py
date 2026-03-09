import asyncio
import time
from typing import Dict, Any, Optional, List

from rag.llm.chat_model import chat_model
from common import get_logger, model_registry, prompt_manager
from rag.entity.clause_tree import ClauseForest, ClauseNode
from repository.cache import cached
from agent.graph_state import HumanDecision
from agent.entity import MedicalEntity
from agent.tools.utils import extract_content

logger = get_logger(__name__)


class FocRetriever:
    """
    Retrieves relevant chunks based on clause forest structure and query analysis.
    """

    def __init__(self, model: Optional[Dict[str, Any]] = None):
        if model is None:
            model_config = model_registry.get_chat_model("qa_reasoner")
            model = model_config.to_dict()

        self.llm = chat_model[model["provider"]](
            model_name=model["model_name"],
            base_url=model["base_url"],
        )

    def _build_foc(self, clause_forest: ClauseForest) -> str:
        def build_tree(node: ClauseNode) -> str:
            header = "#" * (node.level + 2)
            markdown = f"{header} {node.title} [ID:{node.id}]\n"

            for child in node.children:
                markdown += build_tree(child)
            return markdown

        lines = ["## 文档条款结构\n\n"]
        for root in clause_forest.trees.keys():
            lines.append(build_tree(root))
            lines.append("\n")

        return "\n".join(lines)

    def _analyze_query_with_llm(self, query: str, forest_markdown: str) -> Dict[str, Any]:
        prompt = prompt_manager.get(
            "clause_selection_opt",
            clause_structure=forest_markdown,
        )

        try:
            start = time.time()
            reasoning, content, tokens = self.llm.generate(
                messages=[
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"用户问题：{query}\n\n请分析这个问题，并返回最相关的条款ID列表及分析理由。",
                    },
                ]
            )
            logger.warning(f"Time taken to generate: {time.time() - start} seconds")
            result = extract_content(content)
            clause_ids = result.get("relevant_clause_ids", [])

            logger.warning(f"Reasoning: {reasoning}")

            return {
                "relevant_clause_ids": clause_ids,
                "reasoning": result.get("reasoning", reasoning or ""),
                "tokens": tokens,
            }

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}", exc_info=True)
            return {
                "relevant_clause_ids": [],
                "reasoning": f"Analysis failed: {str(e)}",
                "tokens": 0,
            }

    @cached(prefix="foc_llm_analysis", ttl=60 * 60 * 24)
    def retrieve_candidate_chunks(
        self,
        query: str,
        clause_forest: ClauseForest,
    ) -> Dict[str, Any]:
        """
        Select candidate chunk_ids based on query analysis and clause forest structure.
        """
        if not clause_forest or not clause_forest.root.children:
            logger.warning("Empty clause forest provided")
            return {
                "clause_ids": [],
                "reasoning": "Empty clause forest",
                "tokens": 0,
            }

        forest_markdown = self._build_foc(clause_forest)
        logger.debug(f"Forest markdown length: {len(forest_markdown)} chars")

        analysis_result = self._analyze_query_with_llm(query, forest_markdown)
        relevant_clause_ids = analysis_result["relevant_clause_ids"]

        logger.info(f"LLM identified {len(relevant_clause_ids)} relevant clauses: {relevant_clause_ids}")

        return {
            "clause_ids": relevant_clause_ids,
            "reasoning": analysis_result["reasoning"],
            "tokens": analysis_result["tokens"],
        }


async def foc_retrieval(
    entities: List[MedicalEntity], decisions: List[HumanDecision], clause_forest: ClauseForest
) -> Dict[str, Any]:
    """
    Perform FoC (Focus of Care) retrieval based on entities and clause forest.

    Args:
        entities: List of medical entities to search for
        clause_forest: The clause forest structure to search within

    Returns:
        Dictionary containing clause_ids, reasoning, and tokens
    """
    logger.info(f"Performing FoC retrieval with {len(clause_forest.trees)} trees")
    entity_names = set()
    icd10cn_names = set()
    tnm_stages = set()
    for e, decision in zip(entities, decisions):
        entity_names.add(e.term_cn)
        icd10cn_names.add(decision.icd_concept_name + f"（concept code: {decision.icd_concept_code}）")
        tnm_stages.add(decision.tnm_stage)

    query = f"""
      请根据信息判断是否符合主险及附加险的赔付条件：
        诊断：{entity_names}
        ICD10CN：{icd10cn_names}
        TNM分期：{', '.join(tnm_stages)}
    """

    result = await asyncio.to_thread(foc_retriever.retrieve_candidate_chunks, query, clause_forest)
    logger.info(f"FoC retrieval result: {result['clause_ids']}")
    return result


def _create_foc_retriever() -> FocRetriever:
    return FocRetriever()


foc_retriever = _create_foc_retriever()
