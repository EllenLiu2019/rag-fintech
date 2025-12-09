import os
from rag.llm.embedding_model import VoyageEmbed
from common import config, constants
from repository.rdb.models.models import Document as RDBDocument, LLM
from repository.cache.redis_client import cached
import hashlib
from typing import List
import logging

logger = logging.getLogger(__name__)

function_mapping = {
    "Voyage": VoyageEmbed,
}


class EmbeddingService:
    def __init__(self, provider: str, model_name: str):
        self.provider = provider
        self.model_name = model_name
        model_config = next(
            (
                config
                for model in config.EMBEDDING_MODELS
                if model["provider"] == self.provider
                for config in model["models"]
                if config["model_name"] == self.model_name
            ),
            None,
        )
        if not model_config:
            raise ValueError(f"Model config not found for {self.provider} {self.model_name}")
        api_key = os.getenv(self.provider.upper() + constants.API_KEY_SUFFIX)
        self.model = function_mapping[self.provider](key=api_key, model_name=self.model_name)

    def embed_chunks(self, chunks: list[dict], rdb_document: RDBDocument) -> list[dict]:
        """
        Embed chunks with optional batch caching.
        Note: Chunk embedding typically happens during ingestion,
        """
        if not chunks:
            return []

        logger.info(f"Embedding {len(chunks)} chunks using {self.model_name}...")

        texts = [chunk["text"] for chunk in chunks]

        try:
            embeddings, total_tokens = self.model.encode(texts)

            for i, chunk in enumerate(chunks):
                chunk["dense_vector"] = embeddings[i].tolist()

            rdb_document.token_num += total_tokens

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
        return f"embedding:query:{self.model_name}:{text_hash}"

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
            if self.enable_cache and self.cache and self.cache.enabled:
                cache_key = self._get_query_cache_key(text)
                cached_embedding = self.cache.get(cache_key)

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
                    if self.enable_cache and self.cache and self.cache.enabled:
                        cache_key = self._get_query_cache_key(uncached_texts[i])
                        self.cache.set(cache_key, embedding, ttl=3600)

            except Exception as e:
                logger.error(f"Failed to batch embed queries: {e}")
                # Fill with zeros for failed embeddings
                for idx in uncached_indices:
                    if results[idx] is None:
                        results[idx] = []

        return results
