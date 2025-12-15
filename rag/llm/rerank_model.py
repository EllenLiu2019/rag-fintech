"""
Reranker model implementations with factory pattern.
Supports multiple backends: FlagEmbedding (dev), TEI (prod), QWen API, etc.
"""

from abc import ABC, abstractmethod
import numpy as np

from common.exceptions import ModelRateLimitError, RerankError
from common.error_codes import ErrorCodes
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class BaseReranker(ABC):
    """Abstract base class for reranker models."""

    def __init__(self, key: str = None, model_name: str = None, base_url: str = None, **kwargs):
        pass

    @abstractmethod
    def similarity(self, query: str, texts: list[str]) -> tuple[np.ndarray, int]:
        """
        Compute relevance scores between query and texts.

        Args:
            query: The query string
            texts: List of document texts to rank

        Returns:
            Tuple of (scores array, token count)
        """
        raise NotImplementedError("Please implement similarity method!")

    def _estimate_tokens(self, query: str, texts: list[str]) -> int:
        total_chars = len(query) + sum(len(t) for t in texts)
        return total_chars // 4


class FlagEmbeddingRerank(BaseReranker):
    """
    In-process reranker using FlagEmbedding library.
    Best for: Development, small-scale deployment, single-node inference.
    """

    _FACTORY_NAME = "FlagEmbedding"

    def __init__(
        self,
        key: str = None,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        base_url: str = None,
        use_fp16: bool = True,
        device: str = "cuda",
        **kwargs,
    ):
        from FlagEmbedding import FlagReranker

        self.model_name = model_name
        self.model = FlagReranker(model_name, use_fp16=use_fp16, device=device)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ModelRateLimitError),
    )
    def similarity(self, query: str, texts: list[str]) -> tuple[np.ndarray, int]:
        if not texts:
            return np.array([]), 0

        pairs = [[query, text] for text in texts]
        try:
            scores = self.model.compute_score(pairs)
            # Ensure scores is always a list
            if isinstance(scores, (int, float)):
                scores = [scores]
            return np.array(scores), self._estimate_tokens(query, texts)
        except Exception as e:
            raise RerankError(
                message=f"FlagEmbedding rerank failed for query: {query[:50]}...",
                code=ErrorCodes.S_RETRIEVAL_002,
                details={"model": self.model_name, "error": str(e)},
            )


class TEIRerank(BaseReranker):
    """
    HTTP-based reranker using Text Embeddings Inference service.
    Best for: Production deployment, horizontal scaling, resource isolation.

    Requires TEI server running:
        docker run --gpus all -p 8080:80 \\
            ghcr.io/huggingface/text-embeddings-inference:latest \\
            --model-id BAAI/bge-reranker-v2-m3
    """

    _FACTORY_NAME = "TEI"

    def __init__(
        self,
        key: str = None,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        base_url: str = "http://127.0.0.1:8080",
        timeout: int = 30,
        batch_size: int = 32,
        **kwargs,
    ):
        import requests

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self._session = requests.Session()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ModelRateLimitError),
    )
    def similarity(self, query: str, texts: list[str]) -> tuple[np.ndarray, int]:
        if not texts:
            return np.array([]), 0

        scores = np.zeros(len(texts), dtype=float)
        token_count = 0

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            try:
                resp = self._session.post(
                    f"{self.base_url}/rerank",
                    headers={"Content-Type": "application/json"},
                    json={
                        "query": query,
                        "texts": batch_texts,
                        "raw_scores": False,
                        "truncate": True,
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()

                for item in resp.json():
                    scores[item["index"] + i] = item["score"]

                token_count += self._estimate_tokens(query, batch_texts)

            except Exception as e:
                raise RerankError(
                    message=f"TEI rerank failed for query: {query[:50]}...",
                    code=ErrorCodes.S_RETRIEVAL_002,
                    details={"query": query[:50], "error": str(e)},
                )

        return scores, token_count


class QWenRerank(BaseReranker):
    """
    API-based reranker using Alibaba DashScope (Tongyi Qianwen).
    """

    _FACTORY_NAME = "QWen"

    def __init__(
        self,
        key: str,
        model_name: str = "gte-rerank-v2",
        base_url: str = None,
        **kwargs,
    ):
        import dashscope

        self.api_key = key
        self.model_name = model_name if model_name else dashscope.TextReRank.Models.gte_rerank

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ModelRateLimitError),
    )
    def similarity(self, query: str, texts: list[str]) -> tuple[np.ndarray, int]:
        from http import HTTPStatus
        import dashscope

        if not texts:
            return np.array([]), 0

        try:
            resp = dashscope.TextReRank.call(
                api_key=self.api_key,
                model=self.model_name,
                query=query,
                documents=texts,
                top_n=len(texts),
                return_documents=False,
            )

            scores = np.zeros(len(texts), dtype=float)
            if resp.status_code == HTTPStatus.OK:
                for r in resp.output.results:
                    scores[r.index] = r.relevance_score
                return scores, resp.usage.total_tokens
            else:
                raise ValueError(f"QWen API error: {resp.status_code} - {resp.message}")

        except Exception as e:
            raise RerankError(
                message=f"QWen rerank failed for query: {query[:50]}...",
                code=ErrorCodes.S_RETRIEVAL_002,
                details={"query": query[:50], "error": str(e)},
            )


class JinaRerank(BaseReranker):
    """
    API-based reranker using Jina AI.
    Best for: Multilingual reranking, cloud deployment.
    """

    _FACTORY_NAME = "Jina"

    def __init__(
        self,
        key: str,
        model_name: str = "jina-reranker-v3",
        base_url: str = "https://api.jina.ai/v1/rerank",
        **kwargs,
    ):
        self.api_key = key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ModelRateLimitError),
    )
    def similarity(self, query: str, texts: list[str]) -> tuple[np.ndarray, int]:
        if not texts:
            return np.array([]), 0

        import requests

        try:
            resp = requests.post(
                url=self.base_url,
                headers=self.headers,
                json={
                    "model": self.model_name,
                    "query": query,
                    "top_n": len(texts),
                    "documents": texts,
                },
            )
            resp.raise_for_status()

            result = resp.json()
            scores = np.zeros(len(texts), dtype=float)
            for item in result.get("results", []):
                scores[item["index"]] = item["relevance_score"]

            token_count = result.get("usage", {}).get("total_tokens", 0)
            return scores, token_count

        except Exception as e:
            raise RerankError(
                message=f"Jina rerank failed for query: {query[:50]}...",
                code=ErrorCodes.S_RETRIEVAL_002,
                details={"query": query[:50], "error": str(e)},
            )


class CohereRerank(BaseReranker):
    """
    API-based reranker using Cohere.
    Best for: English content, enterprise deployment.
    """

    _FACTORY_NAME = "Cohere"

    def __init__(
        self,
        key: str,
        model_name: str = "rerank-v3.5",
        base_url: str = None,
        **kwargs,
    ):
        import cohere

        self.client = cohere.Client(api_key=key)
        self.model_name = model_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ModelRateLimitError),
    )
    def similarity(self, query: str, texts: list[str]) -> tuple[np.ndarray, int]:
        if not texts:
            return np.array([]), 0

        try:
            resp = self.client.rerank(
                model=self.model_name,
                query=query,
                documents=texts,
                top_n=len(texts),
            )

            scores = np.zeros(len(texts), dtype=float)
            for result in resp.results:
                scores[result.index] = result.relevance_score

            # Cohere doesn't return token count in rerank response
            return scores, self._estimate_tokens(query, texts)

        except Exception as e:
            raise RerankError(
                message=f"Cohere rerank failed for query: {query[:50]}...",
                code=ErrorCodes.S_RETRIEVAL_002,
                details={"query": query[:50], "error": str(e)},
            )


# Provider -> Class mapping
rerank_model = {
    "FlagEmbedding": FlagEmbeddingRerank,
    "TEI": TEIRerank,
    "QWen": QWenRerank,
    "Jina": JinaRerank,
    "Cohere": CohereRerank,
}
