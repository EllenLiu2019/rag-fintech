import json
import re
import time
from typing import Dict, Any, List, Optional, Set

from rag.llm.chat_model import chat_model, VLLm
from common import get_base_config, get_logger, model_registry, prompt_manager
from common.utils import extract_content
from rag.entity.clause_tree import ClauseForest, ClauseNode
from repository.cache import cached

CLAUSE_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_clause_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "List of relevant clause IDs",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief reasoning for the selection",
        },
    },
    "required": ["relevant_clause_ids", "reasoning"],
}

logger = get_logger(__name__)


class FocRetriever:
    """
    Retrieves relevant chunks based on clause forest structure and query analysis.
    """

    def __init__(self, model: Optional[Dict[str, Any]] = None):
        if model is None:
            search_config = get_base_config("search", {})
            model_name = search_config.get("foc_retriever", "qa_lite")
            model_config = model_registry.get_chat_model(model_name)
            model = model_config.to_dict()

        self.llm = chat_model[model["provider"]](
            model_name=model["model_name"],
            base_url=model["base_url"],
        )
        self.temperature = 0

    def _build_foc(self, clause_forest: ClauseForest) -> str:
        def build_tree(node: ClauseNode) -> str:
            header = "#" * (node.level + 2)
            if len(node.chunk_ids) == 0:
                chunk_str = "\\<no chunk\\>"
            elif len(node.chunk_ids) == 1:
                chunk_str = "\\<1 chunk\\>"
            else:
                chunk_str = f"\\<{len(node.chunk_ids)} chunks\\>"
            markdown = f"{header} {node.title} [ID:{node.id}] {chunk_str}\n"

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
            "clause_selection",
            clause_structure=forest_markdown,
        )

        try:
            start = time.time()
            generate_kwargs = {
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": f"用户问题：{query}\n\n请分析这个问题，并返回最相关的条款ID列表及分析理由。",
                    },
                ],
                "temperature": self.temperature,
            }
            if isinstance(self.llm, VLLm):
                generate_kwargs["guided_json"] = CLAUSE_SELECTION_SCHEMA
            reasoning, content, tokens = self.llm.generate(**generate_kwargs)
            logger.info(f"Time taken to generate: {time.time() - start} seconds, tokens: {tokens}")
            result = extract_content(content)
            clause_ids = result.get("relevant_clause_ids", [])

            if not clause_ids:
                match = re.search(r"relevant_clause_ids[\"']?\s*:\s*\[([^\]]+)\]", content)
                if match:
                    clause_ids = [int(x) for x in re.findall(r"\d+", match.group(1))]
                    logger.info(f"Extracted clause_ids via regex fallback: {clause_ids}")

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

    def _get_chunk_ids(
        self,
        clause_forest: ClauseForest,
        clause_ids: List[int],
        include_children: bool = False,
    ) -> Set[str]:

        chunk_ids: set[str] = set()
        selected_nodes: set[ClauseNode] = set()

        for clause_id in clause_ids:
            node = clause_forest.root.find_node_by_id(clause_id)
            if node:
                selected_nodes.add(node)

        if not include_children:
            for node in selected_nodes:
                chunk_ids.update(node.chunk_ids)
            return chunk_ids

        for node in selected_nodes:
            node.get_nodes(
                lambda x: x.chunk_ids if x.chunk_ids else None,
                chunk_ids,
            )

        return chunk_ids

    # @cached(prefix="foc_llm_analysis", ttl=60 * 60 * 24)
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
                "chunk_ids": [],
                "clause_ids": [],
                "reasoning": "Empty clause forest",
                "tokens": 0,
            }

        forest_markdown = self._build_foc(clause_forest)
        logger.debug(f"Forest markdown length: {len(forest_markdown)} chars")

        analysis_result = self._analyze_query_with_llm(query, forest_markdown)
        relevant_clause_ids = analysis_result["relevant_clause_ids"]

        if not relevant_clause_ids:
            logger.info(f"No relevant clauses identified for query: {query[:50]}")
            return {
                "chunk_ids": [],
                "reasoning": analysis_result["reasoning"],
                "tokens": analysis_result["tokens"],
            }

        logger.info(f"LLM identified {len(relevant_clause_ids)} relevant clauses: {relevant_clause_ids}")

        chunk_ids = self._get_chunk_ids(
            clause_forest,
            relevant_clause_ids,
        )

        logger.info(f"Found {len(chunk_ids)} candidate chunks from {len(relevant_clause_ids)} clauses")

        return {
            "chunk_ids": list(chunk_ids),
            "reasoning": analysis_result["reasoning"],
            "tokens": analysis_result["tokens"],
        }


def _create_foc_retriever() -> FocRetriever:
    return FocRetriever()


foc_retriever = _create_foc_retriever()

if __name__ == "__main__":
    from pathlib import Path
    from rag.ingestion.extractor.clause_forest_builder import ClauseForestBuilder
    from rag.entity import RagDocument
    from rag.ingestion.indexing.markdown_splitter import RagMarkdownSplitter

    with open(Path(__file__).parent.parent / "ingestion" / "extractor" / "data" / "policy_base.json", "r") as f:
        document = json.load(f)
    clause_forest_builder = ClauseForestBuilder()
    clause_forest = clause_forest_builder.build(document["pages"])
    rag_document = RagDocument(
        document_id="123",
        pages=document["pages"],
        business_data=document["business_data"],
        confidence={"overall_confidence": 0.95},
        token_num=1000,
        filename="policy_base.json",
        file_size=100,
        content_type="application/json",
        clause_forest=clause_forest,
    )
    splitter = RagMarkdownSplitter()
    splitter.split_document(rag_document)
    foc_retriever = FocRetriever()
    # queries = [
    #     "TNM分期为I期的甲状腺癌是否可以获得主险和附加险同时赔付？",
    #     # "主险和附加险的保障范围有什么区别？",
    # ]
    # for query in queries:
    #     results = foc_retriever.retrieve(
    #         query,
    #         kb_id="default_kb",
    #         query_vector=query_vector,
    #         top_k=10,
    #         clause_forest=rag_document.clause_forest,
    #         search_results=[],
    #     )
