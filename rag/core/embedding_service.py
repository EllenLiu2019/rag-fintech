import os
from rag.llm.embedding_model import embedding_model
from common import constants
from rag.ingestion.document import RagDocument
from common.decorator import cached
import hashlib
from typing import List, Any
import logging
from repository.cache.redis_client import redis_client

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, model: dict[str, Any]):
        api_key = os.getenv(model["provider"].upper() + constants.API_KEY_SUFFIX)
        self.model = embedding_model[model["provider"]](key=api_key, model_name=model["model_name"])

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

        except Exception as e:
            logger.error(f"Failed to embed chunks: {e}")
            raise

    @cached(prefix="embedding", ttl=3600)
    def embed_query(self, text: str) -> list[float]:
        """
        Embed query with caching (TTL: 1 hour).
        Queries are frequently repeated, so caching provides significant benefit.
        """

        try:
            embedding, total_tokens = self.model.encode_queries(text)

            logger.info(f"Embedding query: {text} completed. Total tokens: {total_tokens}")
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise

    def _get_query_cache_key(self, text: str) -> str:
        """Generate cache key for query embedding."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:query:{self.model.model_name}:{text_hash}"

    def embed_queries_batch(self, texts: List[str]) -> List[list[float]]:
        """
        Batch embed queries with caching.

        Args:
            texts: List of query texts

        Returns:
            List of embeddings
        """
        results = []
        uncached_texts = []
        uncached_indices = []

        # Check cache for each query
        for idx, text in enumerate(texts):
            cache_key = self._get_query_cache_key(text)
            cached_embedding = redis_client.get(cache_key)

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
                embeddings, _ = self.model.encode(uncached_texts, input_type="query")

                for i, idx in enumerate(uncached_indices):
                    embedding = embeddings[i].tolist()
                    results[idx] = embedding

                    # Cache the result
                    cache_key = self._get_query_cache_key(uncached_texts[i])
                    redis_client.set(cache_key, embedding, ttl=3600)

            except Exception as e:
                logger.error(f"Failed to batch embed queries: {e}")
                # Fill with zeros for failed embeddings
                for idx in uncached_indices:
                    if results[idx] is None:
                        results[idx] = []

        return results
