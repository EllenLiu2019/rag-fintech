from typing import Optional, Any
import re

from markdown_it import MarkdownIt
from rag.entity import ClauseNode, ClauseForest

from common import get_logger

logger = get_logger(__name__)

TITLE_PATTERNS = [
    {
        "level": 1,
        "pattern": re.compile(r"^(?:#+\s*|\*\*)?(?:.*?保险条款.*?（[^）]*）)"),
    },  # match: 保险条款（...）
    {
        "level": 2,
        "pattern": re.compile(r"^(?:#+\s*|\*\*)?第[一二三四五六七八九十]+部分\s*[\S\s]+"),
    },  # match: ## 第一部分, **第一部分**, 第一部分
    {
        "level": 3,
        "pattern": re.compile(r"^(?:#+\s*|\*\*)?(?:第[一二三四五六七八九十]+条|[一二三四五六七八九十]+、)\s*[\S\s]+"),
    },  # match: ### 第一条, **第一条**, 第一条, 一、
    {
        "level": 4,
        "pattern": re.compile(r"^(?:#+\s*|\*\*)?\（[一二三四五六七八九十]+\）\s*[\S\s]+"),
    },  # match: ## （一）, **（一）**, （一）
]


class ClauseForestBuilder:
    """
    Builds clause forest structure from documents.
    """

    def __init__(self) -> None:
        self.md = MarkdownIt().disable("image").enable("table")
        self.clause_forest: ClauseForest = ClauseForest()
        self.stack: list[ClauseNode] = [self.clause_forest.root]

    def build(self, documents: list[dict[str, Any]]) -> ClauseForest:

        page_number = None
        for doc_idx, document in enumerate(documents):
            metadata = document.get("metadata", {})
            page_number = metadata.get("page_number", doc_idx + 1)
            text = document.get("text", "")

            if not text:
                continue

            tokens = list(filter(lambda x: x.type in ["inline", "html_block"], self.md.parse(text)))

            for token in tokens:
                current_level, title = self._parse_title(token.content)

                # new clause started
                if current_level == 1:
                    self.clause_forest.clause_count += 1

                # not matched, belongs to the node on top of the stack
                if self.clause_forest.clause_count > 0:
                    if current_level is None:
                        self.stack[-1].content += token.content
                        self.stack[-1].pages.add(page_number)
                        continue

                    # title matched, if level <= stack top level, pop the stack until the level is less than the new level
                    while current_level <= self.stack[-1].level:
                        self.stack.pop()

                    # add the new node as the child of the top of the stack
                    # and push the new node to the stack, which will be the parent of the next node
                    self._add_clause_node(title, current_level, page_number)

        self._update_last_tree_node(page_number)
        self._ensure_tree_structure()

        logger.info(f"Extracted clause forest with {self.clause_forest.clause_count} clauses")
        return self.clause_forest

    def _parse_title(self, text: str) -> tuple[Optional[int], Optional[str]]:
        for pattern in TITLE_PATTERNS:
            match = pattern["pattern"].match(text)
            level = pattern["level"]
            if match:
                return level, match.group(0)
        return None, None

    def _add_clause_node(self, title: str, level: int, page_number: int) -> None:
        self.clause_forest.node_count += 1

        new_node = ClauseNode(
            id=self.clause_forest.node_count,
            title=title,
            level=level,
            type="leaf",
            pages={page_number},  # Use set literal instead of set() constructor
            parent=self.stack[-1],
        )
        self.stack[-1].children.append(new_node)
        # if the node has children, it is an index node
        self.stack[-1].type = "index"
        self.stack.append(new_node)

        if level == 1:
            self._update_last_tree_node(page_number - 1)
            self.clause_forest.trees[new_node] = (page_number, -1)

        logger.debug(f"Added clause node: {title} (level {level}, parent: {self.stack[-1].title})")

    def _update_last_tree_node(self, page_number: int) -> None:
        if not self.clause_forest.trees:
            return

        node, (start_page, end_page) = list(self.clause_forest.trees.items())[-1]
        if end_page == -1:
            self.clause_forest.trees[node] = (start_page, page_number)

    def _ensure_tree_structure(self) -> None:
        removed_nodes = []
        for node in list(self.clause_forest.root.children):
            if node.level == 1 and not node.children:
                logger.debug(f"Level 1 node has no children and no content, removing: {node.title}")
                self.clause_forest.trees.pop(node, None)
                self.clause_forest.node_count -= 1
                self.clause_forest.clause_count -= 1
                removed_nodes.append(node.id)

        self.clause_forest.root.children = [
            node for node in self.clause_forest.root.children if node.id not in removed_nodes
        ]


if __name__ == "__main__":
    import json
    from pathlib import Path

    with open(Path(__file__).parent / "data" / "policy_base.json", "r") as f:
        document = json.load(f)
    clause_forest_builder = ClauseForestBuilder()
    clause_forest = clause_forest_builder.build(document["pages"])
    for node in clause_forest.get_forest():
        print(node)
    # forest = clause_forest.serialize()
    # clause_forest_2 = ClauseForest.deserialize(forest)
    # print(clause_forest.root.to_dict() == clause_forest_2.root.to_dict())
    # for node in clause_forest_2.get_forest():
    #     print(node)
