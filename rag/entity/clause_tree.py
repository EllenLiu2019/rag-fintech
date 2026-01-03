from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import Optional, Any, List, Callable, Literal, Set

from common import get_logger

logger = get_logger(__name__)


@dataclass
class ClauseNode:
    """
    Represents a node in the clause tree structure.
    """

    id: int = 0  # Node ID
    title: str = ""  # Clause title
    level: int = 0  # Depth level in the tree (0 = root)
    type: Literal["index", "leaf"] = "leaf"
    content: str = ""
    pages: set[int] = field(default_factory=set)
    parent: Optional["ClauseNode"] = None
    children: list["ClauseNode"] = field(default_factory=list)
    chunk_ids: set[str] = field(default_factory=set)

    def __hash__(self):
        return hash(self.id)

    @classmethod
    def build_tree(
        cls,
        node_data: dict[str, Any],
        parent: Optional["ClauseNode"],
    ) -> "ClauseNode":
        node = cls(
            id=int(node_data["id"]),
            title=node_data["title"],
            level=int(node_data["level"]),
            type=node_data["type"],
            content=node_data["content"],
            pages=set(node_data["pages"]),
            chunk_ids=set(node_data["chunk_ids"]),
            parent=parent,
            children=[],
        )
        for child_data in node_data["children"]:
            child = cls.build_tree(child_data, node)
            node.children.append(child)
        return node

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "type": self.type,
            "content": self.content,
            "pages": sorted(self.pages),
            "parent": self.parent.id if self.parent else None,
            "children": [child.to_dict() for child in self.children] if self.children else [],
            "chunk_ids": list(self.chunk_ids),
        }

    def to_synopsis_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "chunk_count": len(self.chunk_ids),
        }

    def reverse_find_node(self, id: int) -> Optional["ClauseNode"]:
        if self.id == id:
            return self

        for idx in reversed(range(len(self.children))):
            child = self.children[idx]
            if child.id > id:
                continue
            return child.reverse_find_node(id)
        return None

    def build_clause_path(self) -> str:
        """Build the full path from root to node (e.g., "1.8.9.12")."""
        path_parts = []
        current = self

        # Traverse up to root to build path
        while current and current.id != 0:
            path_parts.insert(0, current.id)
            current = current.parent

        return ".".join(str(p) for p in path_parts) if path_parts else str(self.id)

    def get_nodes(
        self,
        filter_func: Callable,
        result: Set[Any],
        should_stop: Optional[Callable[[Any], bool]] = None,
    ) -> None:
        if self.id != 0:
            matched_node = filter_func(self)
            if matched_node:
                if isinstance(matched_node, (AbstractSet, list, tuple)):
                    result.update(matched_node)
                else:
                    result.add(matched_node)
        for child in self.children:
            if should_stop and should_stop(child):
                break
            child.get_nodes(filter_func, result, should_stop)

    def update_chunk_ids(self, chunk_id: str) -> None:
        try:
            self.chunk_ids.add(chunk_id)
        except Exception as e:
            logger.error(f"Failed to update chunk ids:{chunk_id} for node: {self.id}, exc_info: {e}")


@dataclass
class ClauseForest:
    """
    Clause forest structure.
    """

    root: ClauseNode = field(default_factory=lambda: ClauseNode(id=0, title="Document", level=0))
    trees: dict[ClauseNode, tuple[int, int]] = field(default_factory=dict[ClauseNode, tuple[int, int]])
    clause_count: int = 0
    node_count: int = 0

    def __hash__(self):
        trees_data = []
        for node, (start_page, end_page) in sorted(self.trees.items(), key=lambda x: x[0].id):
            trees_data.append((node.id, start_page, end_page))

        node_ids = set()
        self.root.get_nodes(lambda x: x.id, node_ids)
        sorted_node_ids = sorted(node_ids)

        hash_data = (tuple(trees_data), tuple(sorted_node_ids), self.clause_count, self.node_count)

        return hash(hash_data)

    def __str__(self):
        trees_data = []
        for node, (start_page, end_page) in sorted(self.trees.items(), key=lambda x: x[0].id):
            trees_data.append(f"{node.id}:{start_page}-{end_page}")

        node_ids = set()
        self.root.get_nodes(lambda x: x.id, node_ids)
        sorted_node_ids = sorted(node_ids)

        return f"ClauseForest(nodes={sorted_node_ids},trees={trees_data},clause_count={self.clause_count},node_count={self.node_count})"

    def get_forest(self) -> List[dict[str, Any]]:
        result: List[dict[str, Any]] = []
        self.root.get_nodes(lambda x: x.to_synopsis_dict(), result)
        return result

    def serialize(self) -> dict[str, Any]:
        trees_dict: dict[int, list[int]] = {}
        for node, (start_page, end_page) in self.trees.items():
            trees_dict[node.id] = [start_page, end_page]
        return {
            "root": self.root.to_dict(),
            "trees": trees_dict,
            "clause_count": self.clause_count,
            "node_count": self.node_count,
        }

    @classmethod
    def deserialize(cls, data: dict[str, Any]) -> Optional["ClauseForest"]:
        if not data:
            return None

        root = ClauseNode.build_tree(data["root"], None)

        trees: dict[ClauseNode, tuple[int, int]] = {}
        for node_id, (start_page, end_page) in data["trees"].items():
            if not isinstance(node_id, int):
                node_id = int(node_id)
            node = root.reverse_find_node(node_id)
            if node:
                trees[node] = (start_page, end_page)

        return cls(
            root=root,
            trees=trees,
            clause_count=data["clause_count"],
            node_count=data["node_count"],
        )
