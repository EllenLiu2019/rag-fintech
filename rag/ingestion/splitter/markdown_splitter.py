from typing import Any
import re
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document as LlamaDocument
from rag.ingestion.document import RagDocument
from rag.ingestion.splitter.base import BaseSplitter
import logging

logger = logging.getLogger(__name__)

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
            return []

        marked_text = "\n\n".join(marked_text_parts)

        llama_doc = LlamaDocument(text=marked_text, metadata=doc_metadata)

        nodes = self.parser.get_nodes_from_documents([llama_doc])

        chunks = []
        for node in nodes:
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

            chunk = {
                "chunk_id": node.node_id,
                "text": cleaned_text,
                "metadata": node.metadata,
                "prev_chunk": node.prev_node.node_id if node.prev_node else None,
                "next_chunk": node.next_node.node_id if node.next_node else None,
            }
            logger.info(
                f"chunk_id: {node.node_id}, "
                f"text: {cleaned_text}, "
                f"page_number: {node.metadata['page_number']}, "
                f"prev_chunk: {node.prev_node.node_id if node.prev_node else None}, "
                f"next_chunk: {node.next_node.node_id if node.next_node else None},"
            )

            chunks.append(chunk)

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
