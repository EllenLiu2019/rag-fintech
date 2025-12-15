import os
from typing import Any, List, Union
import numpy as np
from rag.llm.rerank_model import rerank_model
from common import constants, get_model_registry
from common.log_utils import get_logger
from common.exceptions import RerankError, ModelNotFoundError
from common.error_codes import ErrorCodes

logger = get_logger(__name__)


class Reranker:
    def __init__(self, model: dict[str, Any]):
        provider = model["provider"]
        if provider not in rerank_model:
            available = list(rerank_model.keys())
            raise ModelNotFoundError(
                message=f"Unknown reranker provider: {provider}",
                code=ErrorCodes.L_MODEL_001,
                details={"provider": provider, "available": available},
            )

        # Get API key if needed (for API-based providers)
        api_key = os.getenv(provider.upper() + constants.API_KEY_SUFFIX)

        self.model = rerank_model[provider](
            key=api_key,
            model_name=model["model_name"],
            base_url=model["base_url"],
        )
        self.provider = provider
        self.model_name = model["model_name"]
        logger.info(f"Initialized Reranker with {provider}/{model['model_name']}")

    def process(
        self,
        query: str,
        documents: Union[List[dict[str, Any]], List[List[dict[str, Any]]]],
        top_k: int = None,
        text_key: str = "text",
    ) -> list[dict[str, Any]]:
        if not documents:
            return []

        if not query or not query.strip():
            logger.warning("Empty query provided for reranking")
            return documents[:top_k] if top_k else documents

        # Flatten the documents list if it is a list of lists
        flat_documents = []
        if isinstance(documents, list) and isinstance(documents[0], list):
            for doc_list in documents:
                flat_documents.extend(doc_list)
        else:
            flat_documents = documents

        unique_docs = list({doc["id"]: doc for doc in flat_documents}.values())
        texts = [doc.get(text_key, "") for doc in unique_docs]

        # Compute relevance scores
        try:
            scores, token_count = self.model.similarity(query, texts)
            logger.info(f"Reranked {len(unique_docs)} docs, tokens used: {token_count}")
        except Exception as e:
            raise RerankError(
                message="Failed to compute rerank scores",
                code=ErrorCodes.S_RETRIEVAL_002,
                details={
                    "query": query[:50],
                    "document_count": len(unique_docs),
                    "provider": self.provider,
                    "error": str(e),
                },
            ) from e

        # Add scores to documents and sort
        scored_docs = []
        for doc_idx, doc in enumerate(unique_docs):
            doc_with_score = doc.copy()
            doc_with_score["rerank_score"] = float(scores[doc_idx])
            scored_docs.append(doc_with_score)

        # Sort by score descending
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            return scored_docs[:top_k]

        return scored_docs

    def compute_scores(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.array([])

        scores, _ = self.model.similarity(query, texts)
        return scores


# Module-level singleton (eager loading, Java/Spring style)
# Initialized at import time to catch configuration errors early
def _create_reranker() -> Reranker:
    """Create reranker instance at module load time."""
    registry = get_model_registry()
    model_config = registry.get_reranker_model()  # Uses "default" config
    return Reranker(model=model_config.to_dict())


reranker = _create_reranker()
