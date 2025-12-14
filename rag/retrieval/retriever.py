from typing import Optional, Dict, Any, List, Literal

from rag.core.embedding_service import EmbeddingService
from rag.core.doc_service import DocumentService
from repository.rdb.models.models import LLM
from repository.vector.milvus_client import VectorStoreClient
from repository.cache.redis_client import cached
from rag.retrieval.reranker import reranker
from common import get_logger
from rag.retrieval.pre_optimizer import query_optimizer

logger = get_logger(__name__)

SELECT_FIELDS = ["id", "doc_id", "text", "page_number", "prev_chunk", "next_chunk", "business_data", "upload_time"]


class Retriever:
    """
    Main retriever with support for both simple dense search and hybrid search.
    """

    def __init__(self):
        self.document_service = DocumentService()
        self.vector_store = VectorStoreClient()

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

        optimized_queries = query_optimizer.optimize(query)["optimized_queries"]

        llm: LLM = self.document_service.get_embedding_model(kb_id)
        embedder = EmbeddingService(model=llm.to_dict())

        if len(optimized_queries) > 1:
            query_vectors = embedder.embed_queries_batch(optimized_queries)
        else:
            query_vectors = [embedder.embed_query(optimized_queries[0])]

        if mode == "hybrid":
            results = self._hybrid_search(optimized_queries, kb_id, top_k, query_vectors, filters)
        elif mode == "dense":
            results = self._dense_search(query, kb_id, top_k, query_vectors, filters)
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
        query_vectors: List[List[float]],
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector search."""

        try:
            results = self.vector_store.search(
                selectFields=SELECT_FIELDS,
                query_vectors=query_vectors,
                limit=top_k,
                indexNames="rag_fintech",
                knowledgebaseIds=[kb_id],
                filters=filters,
            )

            results = reranker.process(query, results, top_k)

            return self._get_relevant_chunks(results, kb_id)

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []

    @cached(prefix="hybrid_search", ttl=1800)
    def _hybrid_search(
        self,
        optimized_queries: List[str],
        kb_id: str,
        top_k: int,
        query_vectors: List[List[float]],
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Sparse vector search."""

        try:
            results = self.vector_store.hybrid_search(
                selectFields=SELECT_FIELDS,
                optimized_queries=optimized_queries,
                query_vectors=query_vectors,
                limit=top_k,
                indexNames="rag_fintech",
                knowledgebaseIds=[kb_id],
                filters=filters,
            )

            query = optimized_queries[0]
            results = reranker.process(query, results, top_k)

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


def _create_retriever() -> Retriever:
    retriever = Retriever()

    logger.info("Initialized Retriever singleton")
    return retriever


retriever = _create_retriever()
