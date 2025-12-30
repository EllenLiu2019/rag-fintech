from typing import Optional, Dict, Any, List, Literal, Union

from rag.core import embedder, StorageService
from repository.vector import vector_store
from repository.cache import cached
from rag.retrieval.foc_retriever import foc_retriever
from rag.retrieval.reranker import reranker
from common import get_logger
from rag.retrieval.pre_optimizer import query_optimizer
from common.exceptions import (
    RetrievalError,
    RerankError,
    EmbeddingError,
    VectorStoreError,
    ConnectionError,
)
from common.error_codes import ErrorCodes
from common.constants import VECTOR_RETRIEVE_FIELDS

logger = get_logger(__name__)


class Retriever:
    """
    Main retriever with support for both simple dense search and hybrid search.
    """

    # @cached(prefix="search", ttl=1800)
    def search(
        self,
        query: str,
        kb_id: str = "default_kb",
        top_k: int = 5,
        filters: Optional[Dict] = None,
        mode: Literal["dense", "hybrid"] = "dense",
        opt_mode: Literal["unified", "hyde", "multi"] = "unified",
        **params: Any,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:

        logger.info(f"Searching for: '{query}' in kb: {kb_id} (mode: {mode})")

        if not query or not query.strip():
            logger.warning("Empty query provided, returning empty results")
            return []

        try:
            optimization_result = query_optimizer.optimize(query, mode=opt_mode)
            optimized_queries = optimization_result["optimized_queries"]
        except Exception as e:
            raise RetrievalError(
                message="Failed to optimize query",
                code=ErrorCodes.S_RETRIEVAL_001,
                details={"query": query, "error": str(e)},
            ) from e

        try:
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

        foc_enhance = params.pop("foc_enhance", False)
        if foc_enhance:
            doc_id = filters.get("doc_id")
            clause_forest = StorageService.get_clause_forest(doc_id)
            if clause_forest:
                results, foc_markdown, foc_data = foc_retriever.retrieve(
                    query, kb_id, query_vectors, top_k, clause_forest, results[0]
                )

        if opt_mode == "multi":
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
        elif not foc_enhance:
            results = results[0]

        if params.pop("get_relevant_chunks", False):
            try:
                results = self._get_relevant_chunks(results, kb_id)
            except Exception as e:
                logger.warning(f"Failed to get relevant chunks: {e}", exc_info=True)

        logger.info(f"Found {len(results) if results else 0} results.")

        return {
            "results": results,
            "foc_data": foc_data if foc_enhance else None,
            "foc_markdown": foc_markdown if foc_enhance else None,
            "query_to_use": optimization_result["query_to_use"],
            "snomed_entities": optimization_result.get("snomed_entities", {}),
        }

    # @cached(prefix="dense_search", ttl=1800)
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
            return vector_store.search(
                selectFields=VECTOR_RETRIEVE_FIELDS,
                query_vectors=query_vectors,
                limit=top_k,
                knowledgebaseIds=[kb_id],
                filters=filters,
            )
        except (VectorStoreError, ConnectionError):
            raise
        except Exception as e:
            raise VectorStoreError(
                message="Vector store search failed",
                code=ErrorCodes.R_VECTOR_002,
                details={"kb_id": kb_id, "top_k": top_k, "error": str(e)},
            ) from e

    # @cached(prefix="hybrid_search", ttl=1800)
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
            return vector_store.hybrid_search(
                selectFields=VECTOR_RETRIEVE_FIELDS,
                optimized_queries=optimized_queries,
                query_vectors=query_vectors,
                limit=top_k,
                knowledgebaseIds=[kb_id],
                filters=filters,
            )
        except (VectorStoreError, ConnectionError):
            raise
        except Exception as e:
            raise VectorStoreError(
                message="Vector store hybrid search failed",
                code=ErrorCodes.R_VECTOR_002,
                details={"kb_id": kb_id, "top_k": top_k, "error": str(e)},
            ) from e

    def _get_relevant_chunks(self, results: List[Dict[str, Any]], kb_id: str) -> List[Dict[str, Any]]:
        """Get relevant chunks from the vector store."""
        for result in results:
            prev_chunk_id = result.get("prev_chunk")
            next_chunk_id = result.get("next_chunk")
            if prev_chunk_id != "":
                result["prev_chunk_text"] = vector_store.get(prev_chunk_id, [kb_id])
            if next_chunk_id != "":
                result["next_chunk_text"] = vector_store.get(next_chunk_id, [kb_id])

        return results


def _create_retriever() -> Retriever:
    return Retriever()


retriever = _create_retriever()
