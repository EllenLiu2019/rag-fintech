import os
import logging
from rag.llm.embedding_model import VoyageEmbed

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self.provider = "voyage"

        self.api_key = os.environ.get("VOYAGE_API_KEY")
        self.model_name = "voyage-3-lite"  # voyage-3-large

        if self.provider == "voyage":
            self.model = VoyageEmbed(key=self.api_key, model_name=self.model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        if not chunks:
            return []

        logger.info(f"Embedding {len(chunks)} chunks using {self.provider}...")

        texts = [chunk["text"] for chunk in chunks]

        try:
            embeddings, total_tokens = self.model.encode(texts)

            for i, chunk in enumerate(chunks):
                chunk["dense_vector"] = embeddings[i].tolist()

            logger.info(f"Embedding completed. Total tokens: {total_tokens}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to embed chunks: {e}")
            raise

    def embed_query(self, text: str) -> list[float]:
        try:
            embedding, _ = self.model.encode_queries(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise
