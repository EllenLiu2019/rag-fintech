from abc import ABC
import os
from typing import List, Dict, Optional, AsyncIterator, Any
from openai import OpenAI, AsyncOpenAI
from common import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from common.exceptions import ModelRateLimitError

logger = get_logger(__name__)


class LLM(ABC):
    def __init__(self, api_key: str, model_name: str, base_url: str = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ModelRateLimitError),
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
        # parameter validation
        if messages is None and prompt is None:
            raise ValueError("Either 'messages' or 'prompt' must be provided")

        # build messages
        if messages is not None:
            # directly use messages format (recommended)
            final_messages = messages
        else:
            # backward compatible: convert prompt to messages format
            final_messages = [{"role": "user", "content": prompt}]

        # build request parameters
        params = {
            "model": self.model_name,
            "messages": final_messages,
            "temperature": temperature,
            "stream": False,
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens

        response = self.client.chat.completions.create(**params)
        message = response.choices[0].message
        reasoning_content = getattr(message, "reasoning_content", None)
        return (
            reasoning_content,
            message.content,
            response.usage.total_tokens,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ModelRateLimitError),
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
        retry=retry_if_exception_type(ModelRateLimitError),
    )
    async def stream_generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
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


class Google(LLM):
    def __init__(self, model_name: str, base_url: str = "https://generativelanguage.googleapis.com/v1beta/"):
        super().__init__(
            api_key=os.getenv("GEMINI_API_KEY"),
            model_name=model_name,
            base_url=base_url,
        )


# Provider -> Class mapping
chat_model = {
    "DeepSeek": DeepSeek,
    "Google": Google,
}
