from rag.core.embedding_service import EmbeddingService
from repository.vector.milvus_client import VectorStoreClient
from repository.vector.doc_store_client import MatchDenseExpr
from repository.cache.redis_client import get_cache
import logging
import hashlib
from typing import Optional, Dict, Any, List, Literal

logger = logging.getLogger(__name__)


class Retriever:
    """
    Main retriever with support for both simple dense search and hybrid search.
    """

    def __init__(
        self,
        mode: Literal["dense", "hybrid"] = "dense",
        enable_reranking: bool = False,
        fusion_method: Literal["rrf", "weighted"] = "rrf",
        enable_cache: bool = True,
    ):
        """
        Initialize retriever.

        Args:
            mode: Retrieval mode - "dense" (default) or "hybrid"
            enable_reranking: Enable cross-encoder re-ranking (hybrid mode only)
            fusion_method: Fusion method for hybrid search ("rrf" or "weighted")
            enable_cache: Enable result caching
        """
        self.mode = mode
        self.enable_cache = enable_cache
        self.embedder = EmbeddingService(enable_cache=enable_cache)
        self.vector_store = VectorStoreClient()
        self.cache = get_cache() if enable_cache else None

        # Initialize hybrid retriever if in hybrid mode
        self.hybrid_retriever = None
        if mode == "hybrid":
            try:
                from rag.retrieval.hybrid_retriever import HybridRetriever

                self.hybrid_retriever = HybridRetriever(
                    fusion_method=fusion_method,
                    enable_reranking=enable_reranking,
                )
                logger.info(f"Initialized in hybrid mode with reranking={enable_reranking}")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid retriever: {e}. Falling back to dense mode.")
                self.mode = "dense"

    def search(
        self, query: str, kb_id: str = "default_kb", top_k: int = 5, filters: Optional[Dict] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents with caching.

        Args:
            query: Search query
            kb_id: Knowledge base ID
            top_k: Number of results to return
            filters: Metadata filters
            **kwargs: Additional parameters for hybrid search (dense_weight, sparse_weight)

        Returns:
            List of relevant documents with scores
        """
        logger.info(f"Searching for: '{query}' in kb: {kb_id} (mode: {self.mode})")

        # Try cache first
        cache_key = None
        if self.enable_cache and self.cache and self.cache.enabled:
            cache_key = self._get_search_cache_key(query, kb_id, top_k, filters, **kwargs)
            cached_results = self.cache.get(cache_key)

            if cached_results is not None:
                logger.info(f"Search cache hit for: '{query[:50]}...'")
                return cached_results

        # Cache miss - perform search
        if self.mode == "hybrid" and self.hybrid_retriever:
            results = self.hybrid_retriever.search(query=query, kb_id=kb_id, top_k=top_k, filters=filters, **kwargs)
        else:
            results = self._dense_search(query, kb_id, top_k, filters)

        # Cache results (TTL: 30 minutes)
        if cache_key and self.enable_cache and self.cache and self.cache.enabled:
            self.cache.set(cache_key, results, ttl=1800)
            logger.debug(f"Search results cached for: '{query[:50]}...'")

        return results

    def _dense_search(
        self,
        query: str,
        kb_id: str,
        top_k: int,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector search (original implementation)."""
        # 1. Embed Query
        query_vector = self.embedder.embed_query(query)

        if not query_vector:
            logger.warning("Empty query vector generated.")
            return []

        # 2. Construct Search Expression
        match_expr = MatchDenseExpr(
            vector_column_name="dense_vector",
            embedding_data=query_vector,
            topn=top_k,
            embedding_data_type="float",
            distance_type="COSINE",
            extra_options={"filters": filters or {}},
        )

        # 3. Execute Search
        try:
            results = self.vector_store.search(
                selectFields=["id", "text", "policy_number", "holder_name", "insured_name", "metadata", "doc_id"],
                matchExprs=[match_expr],
                limit=top_k,
                indexNames="rag_fintech",
                knowledgebaseIds=[kb_id],
            )

            logger.info(f"Found {len(results) if results else 0} results.")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

    def index_documents_for_bm25(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for BM25 (hybrid mode only).

        Args:
            documents: List of document chunks
        """
        if self.mode == "hybrid" and self.hybrid_retriever:
            self.hybrid_retriever.index_documents_for_bm25(documents)
            logger.info(f"Indexed {len(documents)} documents for BM25")
        else:
            logger.debug("BM25 indexing skipped (not in hybrid mode)")

    def _get_search_cache_key(self, query: str, kb_id: str, top_k: int, filters: Optional[Dict], **kwargs) -> str:
        """Generate cache key for search results."""
        filters_str = str(sorted(filters.items())) if filters else ""
        kwargs_str = str(sorted(kwargs.items()))
        cache_input = f"{query}|{kb_id}|{top_k}|{filters_str}|{kwargs_str}|{self.mode}"
        cache_hash = hashlib.md5(cache_input.encode()).hexdigest()
        return f"search:results:{cache_hash}"

