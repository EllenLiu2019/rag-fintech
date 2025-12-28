"""
Reranker model implementations with factory pattern.
Supports multiple backends: FlagEmbedding (dev), TEI (prod), QWen API, etc.
"""

from abc import ABC, abstractmethod
import numpy as np

from common import get_logger
from common.exceptions import ModelRateLimitError, ModelTimeoutError, ModelServerError, RerankError
from common.error_codes import ErrorCodes
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

logger = get_logger(__name__)


def _should_not_retry(exception):
    """
    Determine if an exception should NOT be retried.

    Non-retryable exceptions:
    - ValueError: Parameter errors (code issues)
    - NotImplementedError: Code not implemented
    - HTTPError with 4xx status (except 429): Client errors (except rate limit)
    """
    # Non-retryable exception types
    non_retryable_types = (
        ValueError,
        NotImplementedError,
    )

    if isinstance(exception, non_retryable_types):
        return True

    # Check for HTTP errors (requests library)
    try:
        import requests

        if isinstance(exception, requests.exceptions.HTTPError):
            if hasattr(exception, "response") and exception.response is not None:
                status_code = exception.response.status_code
                # 4xx client errors (except 429 rate limit) should not be retried
                if 400 <= status_code < 500 and status_code != 429:
                    return True
    except ImportError:
        pass

    # Check for RerankError with non-retryable status codes
    if isinstance(exception, RerankError):
        # Check if RerankError contains a 4xx status code in details
        if hasattr(exception, "details") and isinstance(exception.details, dict):
            status_code = exception.details.get("status_code")
            if status_code and 400 <= status_code < 500 and status_code != 429:
                return True

    return False


class BaseReranker(ABC):
    """Abstract base class for reranker models."""

    def __init__(self, key: str = None, model_name: str = None, base_url: str = None, **kwargs):
        pass

    @abstractmethod
    def similarity(self, query: str, texts: list[str], top_n: int = 5) -> tuple[np.ndarray, int]:
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
        retry=retry_if_exception(lambda e: not _should_not_retry(e)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying FlagEmbeddingRerank.similarity (attempt {retry_state.attempt_number}/3) "
            f"after error: {retry_state.outcome.exception()}"
        ),
    )
    def similarity(self, query: str, texts: list[str], top_n: int = 5) -> tuple[np.ndarray, int]:
        if not texts:
            return np.array([]), 0

        pairs = [[query, text] for text in texts]
        try:
            scores = self.model.compute_score(pairs)
            # Ensure scores is always a list
            if isinstance(scores, (int, float)):
                scores = [scores]
            return np.array(scores[:top_n]), self._estimate_tokens(query, texts)
        except Exception as e:
            logger.error(f"FlagEmbedding rerank failed: {type(e).__name__}: {e}", exc_info=True)
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
        retry=retry_if_exception(lambda e: not _should_not_retry(e)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying TEIRerank.similarity (attempt {retry_state.attempt_number}/3) "
            f"after error: {retry_state.outcome.exception()}"
        ),
    )
    def similarity(self, query: str, texts: list[str], top_n: int = 5) -> tuple[np.ndarray, int]:
        if not texts:
            return np.array([]), 0

        import requests

        scores = np.zeros(top_n, dtype=float)
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

                scores = scores[:top_n]

                token_count += self._estimate_tokens(query, batch_texts)

            except requests.exceptions.Timeout as e:
                logger.warning(f"TEI rerank timeout (will retry): {e}")
                raise ModelTimeoutError(f"TEI rerank timeout: {str(e)}")
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"TEI rerank connection error (will retry): {e}")
                raise ModelTimeoutError(f"TEI rerank connection error: {str(e)}")
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else None
                logger.warning(f"TEI rerank HTTP error ({status_code}): {e}")
                if status_code == 429:
                    raise ModelRateLimitError(f"TEI rerank rate limit exceeded: {str(e)}")
                elif status_code and 500 <= status_code < 600:
                    raise ModelServerError(
                        message=f"TEI rerank server error {status_code}: {str(e)}", status_code=status_code
                    )
                # 4xx errors (except 429) will be checked by _should_not_retry
                raise RerankError(
                    message=f"TEI rerank failed for query: {query[:50]}...",
                    code=ErrorCodes.S_RETRIEVAL_002,
                    details={"query": query[:50], "status_code": status_code, "error": str(e)},
                )
            except Exception as e:
                logger.error(f"TEI rerank unexpected error: {type(e).__name__}: {e}", exc_info=True)
                raise RerankError(
                    message=f"TEI rerank failed for query: {query[:50]}...",
                    code=ErrorCodes.S_RETRIEVAL_002,
                    details={"query": query[:50], "error": str(e)},
                )

        return scores[:top_n], token_count


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
        retry=retry_if_exception(lambda e: not _should_not_retry(e)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying QWenRerank.similarity (attempt {retry_state.attempt_number}/3) "
            f"after error: {retry_state.outcome.exception()}"
        ),
    )
    def similarity(self, query: str, texts: list[str], top_n: int = 5) -> tuple[np.ndarray, int]:
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
                top_n=top_n,
                return_documents=False,
            )

            scores = np.zeros(top_n, dtype=float)
            if resp.status_code == HTTPStatus.OK:
                for r in resp.output.results:
                    scores[r.index] = r.relevance_score
                return scores, resp.usage.total_tokens
            else:
                status_code = resp.status_code
                logger.warning(f"QWen API error ({status_code}): {resp.message}")
                if status_code == 429:
                    raise ModelRateLimitError(f"QWen rerank rate limit exceeded: {resp.message}")
                elif 500 <= status_code < 600:
                    raise ModelServerError(
                        message=f"QWen rerank server error {status_code}: {resp.message}", status_code=status_code
                    )
                else:
                    # 4xx errors (except 429) will be checked by _should_not_retry
                    raise ValueError(f"QWen API error: {status_code} - {resp.message}")

        except (ModelRateLimitError, ModelTimeoutError, ModelServerError):
            raise
        except ValueError as e:
            logger.error(f"QWen rerank parameter error: {e}")
            raise
        except Exception as e:
            logger.error(f"QWen rerank unexpected error: {type(e).__name__}: {e}", exc_info=True)
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
        retry=retry_if_exception(lambda e: not _should_not_retry(e)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying JinaRerank.similarity (attempt {retry_state.attempt_number}/3) "
            f"after error: {retry_state.outcome.exception()}"
        ),
    )
    def similarity(self, query: str, texts: list[str], top_n: int = 5) -> tuple[np.ndarray, int]:
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
                    "top_n": top_n,
                    "documents": texts,
                },
            )
            resp.raise_for_status()

            result = resp.json()
            scores = np.zeros(top_n, dtype=float)
            for item in result.get("results", []):
                scores[item["index"]] = item["relevance_score"]

            token_count = result.get("usage", {}).get("total_tokens", 0)
            return scores, token_count

        except requests.exceptions.Timeout as e:
            logger.warning(f"Jina rerank timeout (will retry): {e}")
            raise ModelTimeoutError(f"Jina rerank timeout: {str(e)}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Jina rerank connection error (will retry): {e}")
            raise ModelTimeoutError(f"Jina rerank connection error: {str(e)}")
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            logger.warning(f"Jina rerank HTTP error ({status_code}): {e}")
            if status_code == 429:
                raise ModelRateLimitError(f"Jina rerank rate limit exceeded: {str(e)}")
            elif status_code and 500 <= status_code < 600:
                raise ModelServerError(
                    message=f"Jina rerank server error {status_code}: {str(e)}", status_code=status_code
                )
            # 4xx errors (except 429) will be checked by _should_not_retry
            raise RerankError(
                message=f"Jina rerank failed for query: {query[:50]}...",
                code=ErrorCodes.S_RETRIEVAL_002,
                details={"query": query[:50], "status_code": status_code, "error": str(e)},
            )
        except Exception as e:
            logger.error(f"Jina rerank unexpected error: {type(e).__name__}: {e}", exc_info=True)
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
        model_name: str = "rerank-v4.0-pro",
        base_url: str = "https://api.cohere.com/v2/rerank",
        **kwargs,
    ):
        self.api_key = key
        self.model_name = model_name
        self.base_url = base_url

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(lambda e: not _should_not_retry(e)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying CohereRerank.similarity (attempt {retry_state.attempt_number}/3) "
            f"after error: {retry_state.outcome.exception()}"
        ),
    )
    def similarity(self, query: str, texts: list[str], top_n: int = 5) -> tuple[np.ndarray, int]:
        if not texts:
            return np.array([]), 0

        import requests

        try:
            resp = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model_name,
                    "query": query,
                    "documents": texts,
                    "top_n": top_n,
                },
            )

            # Check HTTP status code and raise appropriate exceptions
            resp.raise_for_status()

            # Parse response JSON
            result = resp.json()
            scores = np.zeros(top_n, dtype=float)

            results = result.get("results", [])
            for item in results:
                index = item.get("index", 0)
                relevance_score = item.get("relevance_score", 0.0)
                if index < top_n:
                    scores[index] = relevance_score

            tokens = 0
            meta = result.get("meta", {})
            tokens_info = meta.get("tokens", {})
            if isinstance(tokens_info, dict):
                tokens = tokens_info.get("input_tokens", 0) + tokens_info.get("output_tokens", 0)

            return scores, tokens

        except requests.exceptions.Timeout as e:
            logger.warning(f"Cohere rerank timeout (will retry): {e}")
            raise ModelTimeoutError(f"Cohere rerank timeout: {str(e)}")
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Cohere rerank connection error (will retry): {e}")
            raise ModelTimeoutError(f"Cohere rerank connection error: {str(e)}")
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None

            logger.warning(f"Cohere rerank HTTP error ({status_code}): {str(e)}")

            if status_code == 429:
                raise ModelRateLimitError(f"Cohere rerank rate limit exceeded: {str(e)}")
            elif status_code and 500 <= status_code < 600:
                raise ModelServerError(
                    message=f"Cohere rerank server error {status_code}: {str(e)}", status_code=status_code
                )
            raise RerankError(
                message=f"Cohere rerank failed for query: {query[:50]}...",
                code=ErrorCodes.S_RETRIEVAL_002,
                details={"query": query[:50], "status_code": status_code, "error": str(e)},
            )
        except Exception as e:
            logger.error(f"Cohere rerank unexpected error: {type(e).__name__}: {e}", exc_info=True)
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
