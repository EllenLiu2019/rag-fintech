from dataclasses import dataclass, field
from typing import Optional, Any, List, Callable


@dataclass
class ClauseNode:
    """
    Represents a node in the clause tree structure.
    """

    id: int = 0  # Node ID
    title: str = ""  # Clause title
    level: int = 0  # Depth level in the tree (0 = root)
    content: str = ""
    pages: set[int] = field(default_factory=set)
    parent: Optional["ClauseNode"] = None
    children: list["ClauseNode"] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "content": self.content,
            "pages": self.pages,
            "parent": self.parent.to_dict() if self.parent else None,
            "children": [child.to_dict() for child in self.children],
        }

    def to_dict_with_parent(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "level": self.level,
            "title": self.title,
            "content": self.content,
            "pages": self.pages,
            "parent": self.parent.to_dict_with_parent() if self.parent else None,
        }

    def find_by_id(self, id: int) -> Optional["ClauseNode"]:
        if self.id == id:
            return self
        for child in self.children:
            result = child.find_by_id(id)
            if result:
                return result
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

    def get_tree_nodes(self, func: Callable) -> List[dict[str, Any]]:
        """Get a tree list of all clauses (DFS traversal)."""
        result = []

        def dfs(node: ClauseNode):
            if node.id != 0:
                matched_node = func(node)
                if matched_node:
                    result.append(matched_node)
                elif len(result) > 0:
                    return
            for child in node.children:
                dfs(child)

        dfs(self)
        return result


@dataclass
class ClauseForest:
    """
    Clause forest structure.
    """

    root: ClauseNode = field(default_factory=lambda: ClauseNode(id=0, title="Document", level=0))
    trees: dict[ClauseNode, tuple[int, int]] = field(default_factory=dict[ClauseNode, tuple[int, int]])
    start_page_number: int = 0
    clause_count: int = 0
    node_count: int = 0

    def get_forest_list(self) -> List[dict[str, Any]]:
        return self.root.get_tree_nodes(lambda x: x.to_dict_with_parent())
