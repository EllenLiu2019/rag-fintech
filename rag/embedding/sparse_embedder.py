from typing import Any
import os

from common import get_logger, constants, model_registry
from common.exceptions import EmbeddingError
from common.error_codes import ErrorCodes
from rag.llm.embedding_model import embedding_model
from repository.cache import cached

logger = get_logger(__name__)


class SparseEmbedder:
    def __init__(self, model: dict[str, Any]):
        api_key = os.getenv(model["provider"].upper() + constants.API_KEY_SUFFIX)
        self.provider = model["provider"]
        self.model = embedding_model[model["provider"]](
            key=api_key,
            model_name=model["model_name"],
        )

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:

        if not chunks:
            return []

        logger.info(f"Generating sparse embeddings for {len(chunks)} chunks by {self.provider}...")

        texts = [chunk["text"] for chunk in chunks]

        try:
            embeddings = self.model.encode(texts)

            for chunk, embedding in zip(chunks, embeddings):
                chunk["sparse_vector"] = embedding

            return chunks
        except Exception as e:
            raise EmbeddingError(
                message="Failed to embed chunks",
                code=ErrorCodes.L_EMBEDDING_002,
                details={"text_count": len(texts), "provider": self.provider, "error": str(e)},
            ) from e

    @cached(prefix="sparse_embedding", ttl=3600)
    def embed_queries(self, queries: list[str]) -> list[dict[str, float]]:
        try:
            return self.model.encode(queries)
        except Exception as e:
            raise EmbeddingError(
                message="Failed to embed queries",
                code=ErrorCodes.L_EMBEDDING_002,
                details={"query_count": len(queries), "provider": self.provider, "error": str(e)},
            ) from e


def _create_sparse_embedder(provider: str) -> SparseEmbedder:
    model = model_registry.get_embedding_model(provider)
    return SparseEmbedder(model=model.to_dict())


sparse_embedder = _create_sparse_embedder("milvus")
