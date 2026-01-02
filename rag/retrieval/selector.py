from typing import List, Dict, Any, Tuple

from rag.entity.clause_tree import ClauseForest
from common import get_logger

logger = get_logger(__name__)


def merge_chunks(
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

    relevant_foc = build_relevant_foc(chosen_clause_ids, foc_chunks, clause_forest)
    all_results = foc_chunks + non_foc_chunks

    logger.info(
        f"Found {len(all_results)} chunks in total, including {len(foc_chunks)} FOC chunks and {len(non_foc_chunks)} non-FOC chunks"
    )

    return all_results, relevant_foc, clause_forest.serialize()


def build_relevant_foc(
    chosen_clause_ids: set[int],
    results: List[Dict[str, Any]],
    clause_forest: ClauseForest,
) -> str:
    """
    Build relevant FOC with clause and chosen chunk content;
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
