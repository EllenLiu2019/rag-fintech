from typing import List, Dict, Optional, AsyncIterator, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from common import get_logger
from rag.llm.chat_model import LLM

logger = get_logger(__name__)


class ResilientLLM(LLM):
    """
    A wrapper around multiple LLM instances to provide fallback capabilities.
    Implements 'Retry locally first, then Fallback' strategy.
    """

    def __init__(self, primary: LLM, fallbacks: List[LLM]):
        self.primary = primary
        self.fallbacks = fallbacks
        # We don't call super().__init__ because we don't have a single client/model_name
        # But we implement the interface

    def _create_retry_decorator(self):
        # Retry 3 times with exponential backoff (1s, 2s, 4s)
        # Only retry for specific operational errors (simulate by catching Exception for now)
        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),  # In prod, be specific: (RateLimitError, TimeoutError)
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying LLM call... (attempt {retry_state.attempt_number})"
            ),
        )

    def generate(
        self,
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, str, int]:

        candidates = [self.primary] + self.fallbacks
        last_exception = None

        for i, model in enumerate(candidates):
            try:
                if i > 0:
                    logger.warning(f"Falling back to model {model.model_name} (attempt {i})")

                # Define retry logic specifically for this model call
                @self._create_retry_decorator()
                def _generate_with_retry():
                    return model.generate(
                        messages=messages, prompt=prompt, temperature=temperature, max_tokens=max_tokens
                    )

                return _generate_with_retry()
            except Exception as e:
                logger.error(f"Model {model.model_name} failed after retries: {e}")
                last_exception = e
                continue

        raise last_exception

    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:

        candidates = [self.primary] + self.fallbacks
        last_exception = None

        for i, model in enumerate(candidates):
            try:
                if i > 0:
                    logger.warning(f"Falling back to model {model.model_name} (attempt {i})")

                # We need to iterate to ensure it works, but we can't easily "try" a generator
                # So we just return the generator of the first working model
                # Note: This doesn't catch errors *during* generation, only at start if any
                async for chunk in model.stream_generate(
                    messages=messages, temperature=temperature, max_tokens=max_tokens
                ):
                    yield chunk
                return
            except Exception as e:
                logger.error(f"Model {model.model_name} failed: {e}")
                last_exception = e
                continue

        if last_exception:
            raise last_exception
