from rag.core.embedding_service import EmbeddingService
from rag.ingestion.doc_service import DocumentService
from repository.rdb.models.models import LLM
from repository.vector.milvus_client import VectorStoreClient
import logging
from typing import Optional, Dict, Any, List, Literal
from common.decorator import cached
from rag.retrieval.pre_optimizer import QueryOptimizer

logger = logging.getLogger(__name__)

SELECT_FIELDS = ["id", "doc_id", "text", "page_number", "prev_chunk", "next_chunk", "business_data", "upload_time"]


class Retriever:
    """
    Main retriever with support for both simple dense search and hybrid search.
    """

    def __init__(self, model: dict[str, Any]):
        self.document_service = DocumentService()
        self.vector_store = VectorStoreClient()
        self.query_optimizer = QueryOptimizer(model=model)

    @cached(prefix="search", ttl=1800)
    def search(
        self,
        query: str,
        kb_id: str = "default_kb",
        top_k: int = 5,
        filters: Optional[Dict] = None,
        mode: Literal["dense", "hybrid"] = "dense",
    ) -> List[Dict[str, Any]]:

        logger.info(f"Searching for: '{query}' in kb: {kb_id} (mode: {mode})")

        optimized_query = self.query_optimizer.optimize(query)["optimized_query"]

        llm: LLM = self.document_service.get_embedding_model(kb_id)
        embedder = EmbeddingService(provider=llm.llm_provider, model_name=llm.model_name)
        query_vector = embedder.embed_query(optimized_query)

        if not query_vector:
            logger.warning("Empty query vector generated.")
            return []

        if mode == "hybrid":
            results = self._hybrid_search(optimized_query, kb_id, top_k, query_vector, filters)
        elif mode == "dense":
            results = self._dense_search(optimized_query, kb_id, top_k, query_vector, filters)
        else:
            raise ValueError(f"Invalid retrieval mode: {mode}")

        logger.info(f"Found {len(results) if results else 0} results.")
        return results

    @cached(prefix="dense_search", ttl=1800)
    def _dense_search(
        self,
        query: str,
        kb_id: str,
        top_k: int,
        query_vector: list[float],
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector search."""

        try:
            results = self.vector_store.search(
                selectFields=SELECT_FIELDS,
                query_vector=query_vector,
                limit=top_k,
                indexNames="rag_fintech",
                knowledgebaseIds=[kb_id],
                filters=filters,
            )
            return self._get_relevant_chunks(results, kb_id)

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

    @cached(prefix="hybrid_search", ttl=1800)
    def _hybrid_search(
        self,
        query: str,
        kb_id: str,
        top_k: int,
        query_vector: list[float],
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Sparse vector search."""

        try:
            results = self.vector_store.hybrid_search(
                selectFields=SELECT_FIELDS,
                query=query,
                query_vector=query_vector,
                limit=top_k,
                indexNames="rag_fintech",
                knowledgebaseIds=[kb_id],
                filters=filters,
            )

            return self._get_relevant_chunks(results, kb_id)

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

    def _get_relevant_chunks(self, results: List[Dict[str, Any]], kb_id: str) -> List[Dict[str, Any]]:
        """Get relevant chunks from the vector store."""
        for result in results:
            prev_chunk_id = result.get("prev_chunk")
            next_chunk_id = result.get("next_chunk")
            if prev_chunk_id != "":
                result["prev_chunk_text"] = self.vector_store.get(prev_chunk_id, "rag_fintech", [kb_id])
            if next_chunk_id != "":
                result["next_chunk_text"] = self.vector_store.get(next_chunk_id, "rag_fintech", [kb_id])

        return results
