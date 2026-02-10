from abc import ABC
import os
from typing import List, Dict, Optional, AsyncIterator, Any
from openai import OpenAI, AsyncOpenAI, RateLimitError, APITimeoutError, APIError, AuthenticationError
from common import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
from common.exceptions import ModelRateLimitError, ModelTimeoutError, ModelServerError

logger = get_logger(__name__)


def _should_not_retry(exception):
    """
    Determine if an exception should NOT be retried.

    Non-retryable exceptions:
    - ValueError: Parameter errors (code issues)
    - AuthenticationError: API key errors (configuration issues)
    - NotImplementedError: Code not implemented
    - APIError with 4xx status (except 429): Client errors (except rate limit)
    """
    # Non-retryable exception types
    non_retryable_types = (
        ValueError,
        AuthenticationError,
        NotImplementedError,
    )

    if isinstance(exception, non_retryable_types):
        return True

    # 4xx client errors (except 429 rate limit) should not be retried
    if isinstance(exception, APIError) and hasattr(exception, "status_code"):
        status_code = exception.status_code
        if 400 <= status_code < 500 and status_code != 429:
            return True

    return False


class LLM(ABC):
    def __init__(self, api_key: str, model_name: str, base_url: str = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(lambda e: not _should_not_retry(e)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying LLM.generate (attempt {retry_state.attempt_number}/3) "
            f"after error: {retry_state.outcome.exception()}"
        ),
    )
    def generate(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, str, int]:
        """
        generate answer

        Args:
            messages: Messages format conversation list, format: [{"role": "system", "content": "..."}, ...]
            prompt: Single prompt string (backward compatible, if messages is provided, this parameter is ignored)
            temperature: Generation temperature
            max_tokens: Maximum token number, None means using model default value

        Returns:
            (reasoning, content, tokens)

        Note:
            - Use messages parameter (recommended)
            - If only prompt is provided, it will be automatically converted to messages format (backward compatible)
            - messages and prompt cannot be both None
        """
        if messages is None and prompt is None:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        if messages is not None:
            final_messages = messages
        else:
            final_messages = [{"role": "user", "content": prompt}]

        params = {
            "model": self.model_name,
            "messages": final_messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        try:
            response = self.client.chat.completions.create(**params)
            message = response.choices[0].message
            reasoning_content = getattr(message, "reasoning_content", None)
            return (
                reasoning_content,
                message.content,
                response.usage.total_tokens,
            )
        except RateLimitError as e:
            retry_after = None
            if hasattr(e, "response") and hasattr(e.response, "headers"):
                retry_after = e.response.headers.get("retry-after")
            logger.warning(f"Rate limit error (will retry): {e}")
            raise ModelRateLimitError(
                message=f"Rate limit exceeded: {str(e)}", retry_after=int(retry_after) if retry_after else None
            )
        except APITimeoutError as e:
            logger.warning(f"API timeout (will retry): {e}")
            raise ModelTimeoutError(f"API timeout: {str(e)}")
        except APIError as e:
            status_code = getattr(e, "status_code", None)
            logger.warning(f"API error ({status_code or 'unknown'}): {e}")
            if status_code:
                if status_code == 429:
                    raise ModelRateLimitError(f"Rate limit exceeded: {str(e)}")
                elif 500 <= status_code < 600:
                    raise ModelServerError(message=f"Server error {status_code}: {str(e)}", status_code=status_code)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLM.generate: {type(e).__name__}: {e}", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(lambda e: not _should_not_retry(e)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying LLM.agenerate (attempt {retry_state.attempt_number}/3) "
            f"after error: {retry_state.outcome.exception()}"
        ),
    )
    async def agenerate(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, str, int]:
        if messages is None and prompt is None:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        if messages is not None:
            final_messages = messages
        else:
            final_messages = [{"role": "user", "content": prompt}]

        params = {
            "model": self.model_name,
            "messages": final_messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        try:
            response = await self.async_client.chat.completions.create(**params)
            message = response.choices[0].message
            reasoning_content = getattr(message, "reasoning_content", None)
            return (
                reasoning_content,
                message.content,
                response.usage.total_tokens,
            )
        except RateLimitError as e:
            retry_after = None
            if hasattr(e, "response") and hasattr(e.response, "headers"):
                retry_after = e.response.headers.get("retry-after")
            logger.warning(f"Rate limit error (will retry): {e}")
            raise ModelRateLimitError(
                message=f"Rate limit exceeded: {str(e)}", retry_after=int(retry_after) if retry_after else None
            )
        except APITimeoutError as e:
            logger.warning(f"API timeout (will retry): {e}")
            raise ModelTimeoutError(f"API timeout: {str(e)}")
        except APIError as e:
            status_code = getattr(e, "status_code", None)
            logger.warning(f"API error ({status_code or 'unknown'}): {e}")
            if status_code:
                if status_code == 429:
                    raise ModelRateLimitError(f"Rate limit exceeded: {str(e)}")
                elif 500 <= status_code < 600:
                    raise ModelServerError(message=f"Server error {status_code}: {str(e)}", status_code=status_code)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in LLM.agenerate: {type(e).__name__}: {e}", exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(lambda e: not _should_not_retry(e)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying LLM.stream_generate (attempt {retry_state.attempt_number}/3) "
            f"after error: {retry_state.outcome.exception()}"
        ),
    )
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Stream generation

        Args:
            messages: Conversation list
            temperature: Generation temperature
            max_tokens: Maximum token number

        Yields:
            {
                "type": "reasoning" | "content" | "metadata",
                "content": str,
                "tokens": int
            }
        """
        raise NotImplementedError


class DeepSeek(LLM):
    def __init__(self, model_name: str, base_url: str = "https://api.deepseek.com"):
        super().__init__(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model_name=model_name,
            base_url=base_url,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception(lambda e: not _should_not_retry(e)),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying DeepSeek.stream_generate (attempt {retry_state.attempt_number}/3) "
            f"after error: {retry_state.outcome.exception()}"
        ),
    )
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        params = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        try:
            response = await self.async_client.chat.completions.create(**params)

            total_tokens = 0

            async for chunk in response:
                delta = chunk.choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    yield {"type": "reasoning", "content": delta.reasoning_content, "tokens": 0}

                if hasattr(delta, "content") and delta.content:
                    yield {"type": "content", "content": delta.content, "tokens": 0}

                if hasattr(chunk, "usage") and chunk.usage:
                    total_tokens = chunk.usage.total_tokens

            yield {"type": "metadata", "content": "", "tokens": total_tokens}
        except RateLimitError as e:
            retry_after = None
            if hasattr(e, "response") and hasattr(e.response, "headers"):
                retry_after = e.response.headers.get("retry-after")
            logger.warning(f"Rate limit error (will retry): {e}")
            raise ModelRateLimitError(
                message=f"Rate limit exceeded: {str(e)}", retry_after=int(retry_after) if retry_after else None
            )
        except APITimeoutError as e:
            logger.warning(f"API timeout (will retry): {e}")
            raise ModelTimeoutError(f"API timeout: {str(e)}")
        except APIError as e:
            status_code = getattr(e, "status_code", None)
            logger.warning(f"API error ({status_code or 'unknown'}): {e}")
            if status_code:
                if status_code == 429:
                    raise ModelRateLimitError(f"Rate limit exceeded: {str(e)}")
                elif 500 <= status_code < 600:
                    raise ModelServerError(message=f"Server error {status_code}: {str(e)}", status_code=status_code)
            raise
        except Exception as e:
            logger.error(f"Unexpected error in DeepSeek.stream_generate: {type(e).__name__}: {e}", exc_info=True)
            raise


class Google(LLM):
    def __init__(self, model_name: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta/"):
        super().__init__(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name=model_name,
            base_url=base_url,
        )


class VLLm(LLM):
    def __init__(self, model_name: str, base_url: str):
        super().__init__(
            api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
            model_name=model_name,
            base_url=base_url,
        )


# Provider -> Class mapping
chat_model = {
    "DeepSeek": DeepSeek,
    "Google": Google,
    "OpenAI": VLLm,
}
