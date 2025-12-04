from typing import Any
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.schema import Document as LlamaDocument

from rag.ingestion.document import RagDocument
from rag.ingestion.splitter.base import BaseSplitter


class RagMarkdownSplitter(BaseSplitter):
    """
    Split documents using LlamaIndex's MarkdownNodeParser.
    Ensures metadata from the parent document is inherited by all chunks.
    """

    def __init__(self):
        self.parser = MarkdownNodeParser()

    def split_document(self, doc: RagDocument) -> list[dict[str, Any]]:

        combined_metadata = {
            **doc.metadata,
            "filename": doc.filename,
            "document_id": doc.document_id,
            "content_type": doc.content_type or "text/plain",
        }

        combined_metadata = {k: v for k, v in combined_metadata.items() if v is not None}

        llama_doc = LlamaDocument(
            text=doc.text,
            metadata=combined_metadata,
            excluded_embed_metadata_keys=["document_id", "filename", "content_type"],
            excluded_llm_metadata_keys=["document_id", "filename", "content_type", "embed_metadata"],
        )

        nodes = self.parser.get_nodes_from_documents([llama_doc])

        chunks = []
        for node in nodes:
            # Extract and remove embed_metadata from metadata so it doesn't get stored
            embed_text = node.metadata.pop("embed_metadata", "")

            chunk = {
                "chunk_id": node.node_id,
                "text": node.get_content(metadata_mode="none"),
                "metadata": node.metadata,
            }
            chunks.append(chunk)

        return chunks
