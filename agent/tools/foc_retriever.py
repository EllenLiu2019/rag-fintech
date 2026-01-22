import json
import time
from typing import Dict, Any, Optional

from rag.llm.chat_model import chat_model
from common import get_logger, get_model_registry
from common.prompt_manager import get_prompt_manager
from rag.entity.clause_tree import ClauseForest, ClauseNode
from repository.cache import cached

logger = get_logger(__name__)


class FocRetriever:
    """
    Retrieves relevant chunks based on clause forest structure and query analysis.
    """

    def __init__(self, model: Optional[Dict[str, Any]] = None):
        if model is None:
            registry = get_model_registry()
            model_config = registry.get_chat_model("qa_reasoner")
            model = model_config.to_dict()

        self.llm = chat_model[model["provider"]](
            model_name=model["model_name"],
            base_url=model["base_url"],
        )
        self.prompt_manager = get_prompt_manager()

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
        prompt = self.prompt_manager.get(
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
            logger.info(f"Time taken to generate: {time.time() - start} seconds")
            result = json.loads(content.replace("```json", "").replace("```", ""))
            clause_ids = result.get("relevant_clause_ids", [])

            logger.debug(f"Reasoning: {reasoning}")

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


def _create_foc_retriever() -> FocRetriever:
    return FocRetriever()


foc_retriever = _create_foc_retriever()
