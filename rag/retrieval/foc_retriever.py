import json
import time
from typing import Dict, Any, List, Optional, Set, Tuple

from rag.llm.chat_model import chat_model
from common import get_logger, get_model_registry
from common.prompt_manager import get_prompt_manager
from rag.entity.clause_tree import ClauseForest, ClauseNode
from repository.vector import vector_store
from common.constants import VECTOR_GET_FIELDS
from common.utils import cosine_similarity
from repository.cache import cached

logger = get_logger(__name__)

WEIGHT_FACTOR = 1


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
        self.temperature = 1.0

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

    def _build_foc_by_results(
        self,
        chosen_clause_ids: set[int],
        results: List[Dict[str, Any]],
        clause_forest: ClauseForest,
    ) -> str:
        """
        Build FOC with clause and chosen chunk content;
        Example:
        we have a chosen chunk[chunk_id:123-345-678] with clause_id: 22 and clause_path: 3.10.22;
        which contains the following content:
        ```markdown
        ## 第十条 责任免除\n\n### （一）被保险人故意犯罪 \n\n### （二）试验性治疗
        ```
        then we can build the FOC as follows:
        ## 条款内容：
        ### 主险 [clause_id:3] <1 chunk> [chunk_id:xxx-xxx-xxx]
        #### 第二部分 保障内容 [clause_id:10] <1 chunk> [chunk_id:xxx-xxx-xxx]
        ##### 第十条 责任免除 [clause_id:22] <1 chunk> [chunk_id:123-345-678]
        ####### （一）被保险人故意犯罪 [clause_id:23] <no chunk>
        ####### （二）试验性治疗 [clause_id:24] <no chunk>
        """
        if not results:
            return ""

        lines = ["## 条款内容：\n\n"]
        foc_clause_ids: set[int] = set()
        for result in results:
            for tree_id in result["clause_path"].split("."):
                foc_clause_ids.add(int(tree_id))

        for clause_id in sorted(foc_clause_ids):
            tree_node = clause_forest.root.reverse_find_node(clause_id)
            if tree_node:
                header = "#" * (tree_node.level + 2)
                lines.append(f"{header} {tree_node.title} [clause_id:{tree_node.id}]\n")
                lines.append(f"  - {tree_node.content}\n")
                if tree_node.id not in chosen_clause_ids:
                    continue
                for child in tree_node.children:
                    if not child.chunk_ids:
                        lines.append(f"  - {child.title}\n")

        return "\n".join(lines)

    def _analyze_query_with_llm(self, query: str, forest_markdown: str) -> Dict[str, Any]:
        prompt = self.prompt_manager.get(
            "clause_selection",
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
                ],
                temperature=self.temperature,
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

    def _get_chunk_ids(
        self,
        clause_forest: ClauseForest,
        clause_ids: List[int],
        include_children: bool = False,
    ) -> Set[str]:

        chunk_ids: set[str] = set()
        selected_nodes: set[ClauseNode] = set()

        for clause_id in clause_ids:
            node = clause_forest.root.reverse_find_node(clause_id)
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

    def merge_results(
        self,
        foc_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        clause_forest: ClauseForest,
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        foc_chunks = []
        non_foc_chunks = []

        foc_chunks.extend(foc_results)
        chosen_clause_ids: set[int] = set()
        for foc_result in foc_results:
            chosen_clause_ids.add(int(foc_result.get("clause_id")))

        for result in vector_results:
            clause_id = int(result.get("clause_id"))
            if clause_id != -1 and clause_id not in chosen_clause_ids:  # clause but not in FOC chunks
                foc_chunks.append(result)
                chosen_clause_ids.add(clause_id)
            elif clause_id == -1:  # non-clause
                non_foc_chunks.append(result)

        relevant_foc = self._build_foc_by_results(chosen_clause_ids, foc_chunks, clause_forest)
        all_results = foc_chunks + non_foc_chunks

        logger.info(
            f"Found {len(all_results)} chunks in total, including {len(foc_chunks)} FOC chunks and {len(non_foc_chunks)} non-FOC chunks"
        )

        return all_results, relevant_foc, clause_forest.serialize()


def _create_foc_retriever() -> FocRetriever:
    return FocRetriever()


foc_retriever = _create_foc_retriever()

if __name__ == "__main__":
    from pathlib import Path
    from rag.ingestion.extractor.clause_forest_builder import ClauseForestBuilder
    from rag.entity.document import RagDocument
    from rag.ingestion.splitter.markdown_splitter import RagMarkdownSplitter

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
