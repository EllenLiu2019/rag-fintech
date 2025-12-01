from abc import ABC
import os
from typing import List, Dict, Optional, AsyncIterator, Any
from openai import OpenAI, AsyncOpenAI


class LLM(ABC):
    def __init__(self, api_key: str, model_name: str, base_url: str = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

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
        return (
            response.choices[0].message.reasoning_content,
            response.choices[0].message.content,
            response.usage.total_tokens,
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
    def __init__(self):
        super().__init__(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model_name="deepseek-reasoner",
            base_url="https://api.deepseek.com",
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
