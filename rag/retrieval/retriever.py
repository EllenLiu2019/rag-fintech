from typing import Optional, Dict, Any, List, Literal, Tuple
import asyncio

from rag.embedding import dense_embedder, sparse_embedder
from rag.persistence import PersistentService
from repository.vector import vector_store
from repository.cache import async_cached
from rag.retrieval.selector import merge_chunks
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
from common.constants import VECTOR_RETRIEVE_FIELDS, VECTOR_GET_FIELDS
from rag.entity.clause_tree import ClauseForest

logger = get_logger(__name__)


class Retriever:
    """
    Main retriever with support for both simple dense search and hybrid search.
    """

    async def search(
        self,
        query: str,
        kb_id: str = "default_kb",
        top_k: int = 5,
        filters: Optional[Dict] = None,
        mode: Literal["dense", "hybrid"] = "dense",
        opt_mode: Literal["unified", "hyde", "multi"] = "unified",
        **params: Any,
    ) -> Dict[str, Any]:
        """
        Search for relevant documents using vector retrieval.
        This is now an async function to support parallel embedding operations.
        """
        logger.info(f"Searching for: '{query}' in kb: {kb_id} (mode: {mode})")

        try:
            opt_query = await asyncio.to_thread(query_optimizer.optimize, query, mode=opt_mode)
        except Exception as e:
            raise RetrievalError(
                message="Failed to optimize query",
                code=ErrorCodes.S_RETRIEVAL_001,
                details={"query": query, "error": str(e)},
            ) from e

        foc_enhance = params.pop("foc_enhance", False)
        if foc_enhance:
            doc_id = filters.get("doc_id")
            clause_forest = await PersistentService.aget_clause_forest(doc_id)

            foc_task = asyncio.to_thread(self._retrieve_foc_result, query, kb_id, clause_forest)
            vector_task = self._retrieve_vector_results(opt_query["optimized_queries"], kb_id, top_k, filters, mode)
            foc_results, vector_results = await asyncio.gather(foc_task, vector_task)

            results, relevant_foc, foc_data = merge_chunks(foc_results["chunks"], vector_results[0], clause_forest)
        else:
            vector_results = await self._retrieve_vector_results(
                opt_query["optimized_queries"], kb_id, top_k, filters, mode
            )

            results = await asyncio.to_thread(self._reorder_results, query, vector_results, top_k, opt_mode)

        return {
            "results": results,
            "query_to_use": opt_query["query_to_use"],
            "snomed_entities": opt_query.get("snomed_entities", {}),
            "relevant_foc": relevant_foc if foc_enhance else None,
            "foc_data": foc_data if foc_enhance else None,
        }

    def _retrieve_foc_result(self, query: str, kb_id: str, clause_forest: ClauseForest) -> Dict[str, Any]:
        foc_result = foc_retriever.retrieve_candidate_chunks(query, clause_forest)
        chunks = vector_store.get_bulk(list(foc_result["chunk_ids"]), [kb_id], VECTOR_GET_FIELDS)
        return {
            "chunks": chunks,
            "reasoning": foc_result["reasoning"],
        }

    async def _retrieve_vector_results(
        self,
        optimized_queries: List[str],
        kb_id: str,
        top_k: int,
        filters: Optional[Dict] = None,
        mode: Literal["dense", "hybrid"] = "dense",
    ) -> List[Dict[str, Any]]:

        dense_vectors, sparse_vectors = await self._embed_queries(optimized_queries)

        if mode == "hybrid":
            results = await self._hybrid_search(dense_vectors, sparse_vectors, top_k, kb_id, filters)
        elif mode == "dense":
            results = await self._dense_search(dense_vectors, top_k, kb_id, filters)

        logger.info(f"Found {len(results) if results else 0} results.")

        return results

    @async_cached(prefix="dense_search", ttl=1800)
    async def _dense_search(
        self,
        dense_vectors: List[List[float]],
        top_k: int,
        kb_id: str,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Dense vector search."""

        try:
            return await asyncio.to_thread(
                vector_store.search,
                selectFields=VECTOR_RETRIEVE_FIELDS,
                dense_vectors=dense_vectors,
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

    @async_cached(prefix="hybrid_search", ttl=1800)
    async def _hybrid_search(
        self,
        dense_vectors: List[List[float]],
        sparse_vectors: List[Dict[str, float]],
        top_k: int,
        kb_id: str,
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search (dense + sparse)."""

        try:
            return await asyncio.to_thread(
                vector_store.hybrid_search,
                selectFields=VECTOR_RETRIEVE_FIELDS,
                dense_vectors=dense_vectors,
                sparse_vectors=sparse_vectors,
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

    def _reorder_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
        opt_mode: Literal["unified", "hyde", "multi"] = "unified",
    ) -> List[Dict[str, Any]]:
        if opt_mode == "multi":
            try:
                return reranker.process(query, results, top_k)
            except RerankError:
                raise
            except Exception as e:
                raise RerankError(
                    message="Reranking failed",
                    code=ErrorCodes.S_RETRIEVAL_002,
                    details={"query": query[:50], "result_count": len(results), "error": str(e)},
                ) from e

        return results[0] if results else []

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

    async def _embed_queries(self, queries: List[str]) -> Tuple[List[List[float]], List[Dict[str, float]]]:
        try:
            sparse_task = asyncio.to_thread(sparse_embedder.embed_queries, queries)

            if len(queries) > 1:
                dense_task = asyncio.to_thread(dense_embedder.embed_queries_batch, queries)
            else:

                def _embed_single_and_wrap():
                    return [dense_embedder.embed_query(queries[0])]

                dense_task = asyncio.to_thread(_embed_single_and_wrap)

            dense_vectors, sparse_vectors = await asyncio.gather(dense_task, sparse_task)

            return dense_vectors, sparse_vectors
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                message="Failed to embed queries",
                code=ErrorCodes.L_EMBEDDING_001,
                details={"query_count": len(queries), "error": str(e)},
            ) from e


def _create_retriever() -> Retriever:
    return Retriever()


retriever = _create_retriever()
