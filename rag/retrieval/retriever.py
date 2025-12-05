from xxlimited import Str
from rag.core.embedding_service import EmbeddingService
from repository.vector.milvus_client import VectorStoreClient
from repository.cache.redis_client import get_cache
import logging
import hashlib
from typing import Optional, Dict, Any, List, Literal

logger = logging.getLogger(__name__)

SELECT_FIELDS = ["id", "doc_id", "text", "page_number", "prev_chunk", "next_chunk", "business_data", "upload_time"]


class Retriever:
    """
    Main retriever with support for both simple dense search and hybrid search.
    """

    def __init__(
        self,
        enable_cache: bool = False,
    ):
        """
        Initialize retriever.

        Args:
            enable_cache: Enable result caching
        """

        self.enable_cache = enable_cache
        self.embedder = EmbeddingService(enable_cache=enable_cache)
        self.vector_store = VectorStoreClient()
        self.cache = get_cache() if enable_cache else None

    def search(
        self,
        query: str,
        kb_id: str = "default_kb",
        top_k: int = 5,
        filters: Optional[Dict] = None,
        mode: Literal["dense", "hybrid"] = "dense",
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents with caching.

        Args:
            query: Search query
            kb_id: Knowledge base ID
            top_k: Number of results to return
            filters: Metadata filters
            mode: Retrieval mode - "dense" (default) or "hybrid"

        Returns:
            List of relevant documents with scores
        """
        logger.info(f"Searching for: '{query}' in kb: {kb_id} (mode: {mode})")

        # Try cache first
        # cache_key = None
        # if self.enable_cache and self.cache and self.cache.enabled:
        #     cache_key = self._get_search_cache_key(query, kb_id, top_k, filters, **kwargs)
        #     cached_results = self.cache.get(cache_key)

        #     if cached_results is not None:
        #         logger.info(f"Search cache hit for: '{query[:50]}...'")
        #         return cached_results

        # Cache miss - perform search
        if mode == "hybrid":
            results = self._hybrid_search(query, kb_id, top_k, filters)
        elif mode == "dense":
            results = self._dense_search(query, kb_id, top_k, filters)
        else:
            raise ValueError(f"Invalid retrieval mode: {mode}")

        # Cache results (TTL: 30 minutes)
        # if cache_key and self.enable_cache and self.cache and self.cache.enabled:
        #     self.cache.set(cache_key, results, ttl=1800)
        #     logger.debug(f"Search results cached for: '{query[:50]}...'")

        return results

    def _dense_search(
        self,
        query: str,
        kb_id: str,
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector search."""

        query_vector = self.embedder.embed_query(query)

        if not query_vector:
            logger.warning("Empty query vector generated.")
            return []

        try:
            results = self.vector_store.search(
                selectFields=SELECT_FIELDS,
                query_vector=query_vector,
                limit=top_k,
                indexNames="rag_fintech",
                knowledgebaseIds=[kb_id],
                filters=filters,
            )

            logger.info(f"Found {len(results) if results else 0} results.")
            return self._get_relevant_chunks(results, kb_id)

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

    def _hybrid_search(
        self,
        query: str,
        kb_id: str,
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Sparse vector search."""
        query_vector = self.embedder.embed_query(query)

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

    def _get_relevant_chunks(self, results: List[Dict[str, Any]], kb_id: Str) -> List[Dict[str, Any]]:
        """Get relevant chunks from the vector store."""
        for result in results:
            prev_chunk_id = result.get("prev_chunk")
            next_chunk_id = result.get("next_chunk")
            if prev_chunk_id != "":
                result["prev_chunk_text"] = self.vector_store.get(prev_chunk_id, "rag_fintech", [kb_id])
            if next_chunk_id != "":
                result["next_chunk_text"] = self.vector_store.get(next_chunk_id, "rag_fintech", [kb_id])

        return results

    def _get_search_cache_key(self, query: str, kb_id: str, top_k: int, filters: Optional[Dict], **kwargs) -> str:
        """Generate cache key for search results."""
        filters_str = str(sorted(filters.items())) if filters else ""
        kwargs_str = str(sorted(kwargs.items()))
        cache_input = f"{query}|{kb_id}|{top_k}|{filters_str}|{kwargs_str}|{self.mode}"
        cache_hash = hashlib.md5(cache_input.encode()).hexdigest()
        return f"search:results:{cache_hash}"
