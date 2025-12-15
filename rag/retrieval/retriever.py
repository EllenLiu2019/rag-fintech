from typing import Optional, Dict, Any, List, Literal

from rag.core.embedding_service import EmbeddingService
from rag.core.doc_service import DocumentService
from repository.rdb.models.models import LLM
from repository.vector.milvus_client import VectorStoreClient
from repository.cache.redis_client import cached
from rag.retrieval.reranker import reranker
from common import get_logger
from rag.retrieval.pre_optimizer import query_optimizer
from common.exceptions import (
    RetrievalError,
    RerankError,
    EmbeddingError,
    VectorStoreError,
    ValidationError,
    ConnectionError,
)
from common.error_codes import ErrorCodes

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

        if not query or not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return []

        try:
            optimized_queries = query_optimizer.optimize(query)["optimized_queries"]
        except Exception as e:
            raise RetrievalError(
                message="Failed to optimize query",
                code=ErrorCodes.S_RETRIEVAL_001,
                details={"query": query, "error": str(e)},
            ) from e

        try:
            llm: LLM = self.document_service.get_embedding_model(kb_id)
            embedder = EmbeddingService(model=llm.to_dict())

            if len(optimized_queries) > 1:
                query_vectors = embedder.embed_queries_batch(optimized_queries)
            else:
                query_vectors = [embedder.embed_query(optimized_queries[0])]
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                message="Failed to embed query",
                code=ErrorCodes.L_EMBEDDING_001,
                details={"query": query[:50], "error": str(e)},
            ) from e

        if mode == "hybrid":
            results = self._hybrid_search(optimized_queries, kb_id, top_k, query_vectors, filters)
        elif mode == "dense":
            results = self._dense_search(query, kb_id, top_k, query_vectors, filters)
        else:
            raise ValidationError(
                message=f"Invalid retrieval mode: {mode}",
                code=ErrorCodes.A_VALIDATION_001,
                details={"mode": mode, "valid_modes": ["dense", "hybrid"]},
            )

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
        except (VectorStoreError, ConnectionError):
            # Re-raise storage-related errors as-is (already properly formatted)
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            raise VectorStoreError(
                message="Vector store search failed",
                code=ErrorCodes.R_VECTOR_002,
                details={"kb_id": kb_id, "top_k": top_k, "error": str(e)},
            ) from e

        try:
            results = reranker.process(query, results, top_k)
        except RerankError:
            raise
        except Exception as e:
            raise RerankError(
                message="Reranking failed",
                code=ErrorCodes.S_RETRIEVAL_002,
                details={"query": query[:50], "result_count": len(results), "error": str(e)},
            ) from e

        try:
            return self._get_relevant_chunks(results, kb_id)
        except Exception as e:
            logger.warning(f"Failed to get relevant chunks: {e}", exc_info=True)
            # Return results without context chunks rather than failing
            return results

    @cached(prefix="hybrid_search", ttl=1800)
    def _hybrid_search(
        self,
        optimized_queries: List[str],
        kb_id: str,
        top_k: int,
        query_vectors: List[List[float]],
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search (dense + sparse)."""

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
        except (VectorStoreError, ConnectionError):
            # Re-raise storage-related errors as-is (already properly formatted)
            raise
        except Exception as e:
            # Wrap unexpected exceptions
            raise VectorStoreError(
                message="Vector store hybrid search failed",
                code=ErrorCodes.R_VECTOR_002,
                details={"kb_id": kb_id, "top_k": top_k, "error": str(e)},
            ) from e

        query = optimized_queries[0]
        try:
            results = reranker.process(query, results, top_k)
        except RerankError:
            raise
        except Exception as e:
            raise RerankError(
                message="Reranking failed in hybrid search",
                code=ErrorCodes.S_RETRIEVAL_002,
                details={"query": query[:50], "result_count": len(results), "error": str(e)},
            ) from e

        try:
            return self._get_relevant_chunks(results, kb_id)
        except Exception as e:
            logger.warning(f"Failed to get relevant chunks: {e}", exc_info=True)
            # Return results without context chunks rather than failing
            return results

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
