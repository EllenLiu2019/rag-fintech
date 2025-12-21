import os
import hashlib
from typing import List, Any

from rag.llm.embedding_model import embedding_model
from common import constants, get_logger, get_model_registry
from common.exceptions import EmbeddingError
from common.error_codes import ErrorCodes
from rag.ingestion.document import RagDocument
from repository.cache.redis_client import cached, _get_redis_client

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model: dict[str, Any]):
        api_key = os.getenv(model["provider"].upper() + constants.API_KEY_SUFFIX)
        self.model = embedding_model[model["provider"]](key=api_key, model_name=model["model_name"])
        self.redis_client = _get_redis_client()

    def embed_chunks(self, chunks: list[dict], rag_document: RagDocument) -> list[dict]:
        """
        Embed chunks with optional batch caching.
        Note: Chunk embedding typically happens during ingestion,
        """
        if not chunks:
            return []

        logger.info(f"Embedding {len(chunks)} chunks using {self.model.model_name}...")

        texts = [chunk["text"] for chunk in chunks]

        try:
            embeddings, total_tokens = self.model.encode(texts)

            for i, chunk in enumerate(chunks):
                chunk["dense_vector"] = embeddings[i].tolist()

            rag_document.token_num += total_tokens

            logger.info(f"Embedding completed. Total tokens: {total_tokens}")
            return chunks

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                message="Failed to embed chunks",
                code=ErrorCodes.L_EMBEDDING_001,
                details={"chunk_count": len(chunks), "model": self.model.model_name, "error": str(e)},
            ) from e

    @cached(prefix="embedding", ttl=3600, key_func=lambda self, text: self._cache_key(text))
    def embed_query(self, text: str) -> list[float]:
        """
        Embed query with caching (TTL: 1 hour).
        Queries are frequently repeated, so caching provides significant benefit.
        """

        try:
            embedding, total_tokens = self.model.encode_queries(text)

            logger.info(f"Embedding query: {text} completed. Total tokens: {total_tokens}")
            return embedding.tolist()

        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(
                message="Failed to embed query",
                code=ErrorCodes.L_EMBEDDING_001,
                details={"query": text[:50], "model": self.model.model_name, "error": str(e)},
            ) from e

    def _cache_key(self, text: str) -> str:
        """Generate cache key for query embedding."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"query:{self.model.model_name}:{text_hash}"

    def embed_queries_batch(self, texts: List[str]) -> List[list[float]]:
        """
        Batch embed queries with caching.
        """
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each query
        for idx, text in enumerate(texts):
            if self.redis_client.redis_enabled:
                cache_key = self._cache_key(text)
                cached_embedding = self.redis_client.get(cache_key)

                if cached_embedding is not None:
                    results.append(cached_embedding)
                    continue

            # Mark as uncached
            results.append(None)
            uncached_texts.append(text)
            uncached_indices.append(idx)

        # Batch process uncached queries
        if uncached_texts:
            try:
                embeddings, _ = self.model.encode(uncached_texts)

                for i, idx in enumerate(uncached_indices):
                    embedding = embeddings[i].tolist()
                    results[idx] = embedding

                    # Cache the result
                    if self.redis_client.redis_enabled:
                        cache_key = self._cache_key(uncached_texts[i])
                        self.redis_client.set(cache_key, embedding, ttl=3600)

            except EmbeddingError:
                raise
            except Exception as e:
                # For batch operations, we could either:
                # 1. Raise exception (fail fast)
                # 2. Fill with empty lists (graceful degradation)
                # Current implementation uses graceful degradation
                logger.error(
                    f"Failed to batch embed queries: {e}",
                    extra={"text_count": len(uncached_texts), "model": self.model.model_name},
                )
                # Fill with empty lists for failed embeddings (graceful degradation)
                for idx in uncached_indices:
                    if results[idx] is None:
                        results[idx] = []

        return results


def _create_embedder() -> EmbeddingService:
    registry = get_model_registry()
    model_config = registry.get_embedding_model("dense")
    embedder = EmbeddingService(model=model_config.to_dict())
    logger.info("Initialized embedder singleton")
    return embedder


embedder = _create_embedder()
