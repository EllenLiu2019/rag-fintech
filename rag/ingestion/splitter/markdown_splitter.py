from typing import Any, Optional
import re
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document as LlamaDocument
from rag.entity import RagDocument
from rag.ingestion.splitter.base import BaseSplitter
from common import get_logger
from rag.entity import ClauseNode, ClauseForest

logger = get_logger(__name__)

# Build page markers: insert page markers between pages to track page numbers
PAGE_MARKER = "[PAGE:"


class RagMarkdownSplitter(BaseSplitter):
    """
    Split documents using LlamaIndex's MarkdownNodeParser.
    Ensures metadata from the parent document is inherited by all chunks.
    Tracks page numbers for each chunk based on source pages.
    """

    def __init__(self):
        self.parser = MarkdownNodeParser(
            include_metadata=True,
            include_prev_next_rel=True,
            header_path_separator="/",
            callback_manager=CallbackManager([LlamaDebugHandler()]),
        )

        self.title_matchers = {}
        self.candidates = {}

    def split_document(self, doc: RagDocument) -> list[dict[str, Any]]:

        doc_metadata = {k: v for k, v in doc.to_document_metadata().items() if v is not None}

        # Build text with page markers
        marked_text_parts = []
        curr_page_num = 0
        for page in doc.pages:
            if curr_page_num == 0:
                curr_page_num = page.get("metadata", {}).get("page_number", 0)

            page_num = page.get("metadata", {}).get("page_number", 0)
            page_text = page.get("text", "").strip()
            if page_text:
                marked_text_parts.append(f"{page_text}{PAGE_MARKER}{page_num}]")

        if not marked_text_parts:
            logger.warning("No text content found in document pages")
            return []

        marked_text = "\n\n".join(marked_text_parts)

        llama_doc = LlamaDocument(text=marked_text, metadata=doc_metadata)

        nodes = self.parser.get_nodes_from_documents([llama_doc])

        chunks = []
        for idx, node in enumerate(nodes):
            try:
                chunk_text = node.get_content(metadata_mode="none")

                # Extract page number from chunk text by looking for page markers
                extracted_page_num, page_marker_count = self._extract_page_markers(chunk_text)

                # Remove page markers from chunk text
                cleaned_text = self._remove_page_markers(chunk_text)

                # Add page number to metadata
                # if current chunk has a page number on its own, use it.
                # encounter the last chunk of the page(maybe spans multiple pages), need to prepare for the next chunk of the next page
                # update the current page number for the next chunk
                if extracted_page_num > 0:
                    node.metadata["page_number"] = extracted_page_num
                    # spans one page, so the next chunk should be on the next page
                    curr_page_num = extracted_page_num + page_marker_count
                # if current chunk has no page number on its own
                else:
                    node.metadata["page_number"] = curr_page_num

                # Associate chunk with clause if clause_forest is provided
                clause_node = None
                if doc.clause_forest:
                    try:
                        clause_node = self.find_clause(doc.clause_forest, cleaned_text, node.metadata["page_number"])
                    except Exception as e:
                        logger.warning(
                            f"Failed to find clause for chunk {idx}: {e}",
                            exc_info=True,
                            extra={"chunk_text_preview": cleaned_text[:100] if cleaned_text else ""},
                        )
                        # Continue processing even if clause finding fails
                        clause_node = None

                chunk = {
                    "chunk_id": node.node_id,
                    "text": cleaned_text,
                    "metadata": node.metadata,
                    "prev_chunk": node.prev_node.node_id if node.prev_node else None,
                    "next_chunk": node.next_node.node_id if node.next_node else None,
                    "clause_id": clause_node.id if clause_node else -1,
                    "clause_title": clause_node.title if clause_node else "N/A",
                    "clause_path": clause_node.build_clause_path() if clause_node else "N/A",
                }
            except Exception as e:
                logger.error(
                    f"Failed to process node {idx}: {e}",
                    exc_info=True,
                    extra={"node_id": getattr(node, "node_id", None)},
                )
                # Skip this node and continue with the next one
                continue

            chunks.append(chunk)

        doc.chunk_num += len(chunks)

        return chunks

    def _extract_page_markers(self, text: str) -> tuple[int, int]:
        page_pattern = re.escape(PAGE_MARKER) + r"(\d+)\]"
        page_matches = re.findall(page_pattern, text)

        if not page_matches:
            return -1, 0

        if page_matches:
            try:
                return int(page_matches[0]), len(page_matches)
            except (ValueError, IndexError):
                pass

        return -1, 0

    def _remove_page_markers(self, text: str) -> str:
        page_pattern_remove = re.escape(PAGE_MARKER) + r"\d+\]"
        return re.sub(page_pattern_remove, "", text)

    def find_clause(
        self,
        clause_forest: ClauseForest,
        chunk_text: str,
        page_number: int,
    ) -> Optional[ClauseNode]:
        """
        Find the most relevant clause for a given chunk based on page number and text content.
        """
        forest_root: ClauseNode = clause_forest.root
        if not forest_root or not forest_root.children:
            return None

        if page_number < clause_forest.start_page_number:
            return None

        tree_root: Optional[ClauseNode] = None
        for node, (start_page, end_page) in clause_forest.trees.items():
            if page_number >= start_page and page_number <= end_page:
                tree_root = node
                break
        if not tree_root:
            return None

        candidates = self._get_candidates(tree_root, page_number)

        best_match: Optional[ClauseNode] = None
        best_score = 0

        chunk_text_lines = chunk_text.split("\n\n")
        for candidate in candidates:
            node: ClauseNode = candidate
            score = 0

            # Text matching score
            if node.title:
                title_matcher = self._create_title_matcher(node.title)
                title_match = title_matcher.match(chunk_text)
                if title_match:
                    score += 20  # Title match is strong signal

            # split chunk_text by "\n\n", the result is a whole line of the original text
            # if the line is in node.content, then score += 5
            for line in chunk_text_lines:
                if line in node.content and len(line) > 1:
                    score += 5

            # Prefer higher level clauses (more specific)
            score += node.level * 1 if score > 0 else 0

            if score > best_score:
                best_score = score
                best_match = node

        return best_match

    def _create_title_matcher(self, title: str) -> re.Pattern:
        if title in self.title_matchers:
            return self.title_matchers[title]

        pattern = rf"^(?:#+\s*|\*\*)?{re.escape(title)}"
        matcher = re.compile(pattern)
        self.title_matchers[title] = matcher
        return matcher

    def _get_candidates(self, forest_root: ClauseNode, page_number: int) -> list[ClauseNode]:
        if page_number in self.candidates:
            return self.candidates[page_number]
        candidates = forest_root.get_tree_nodes(lambda x: x if page_number in x.pages else None)
        self.candidates[page_number] = candidates
        return candidates


if __name__ == "__main__":
    import json
    from pathlib import Path
    from rag.ingestion.extractor.clause_forest_builder import ClauseForestBuilder

    with open(Path(__file__).parent.parent / "extractor" / "data" / "policy_base.json", "r") as f:
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
    chunks = splitter.split_document(rag_document)
