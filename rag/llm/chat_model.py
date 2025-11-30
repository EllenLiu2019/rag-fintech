from abc import ABC
import os
import json
from typing import List, Dict, Optional
from openai import OpenAI


class LLM(ABC):
    def __init__(self, api_key: str, model_name: str, base_url: str = None):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def generate(
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

        response = self.client.chat.completions.create(**params)
        return (
            response.choices[0].message.reasoning_content,
            response.choices[0].message.content,
            response.usage.total_tokens,
        )


class DeepSeek(LLM):
    def __init__(self):
        super().__init__(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            model_name="deepseek-reasoner",
            base_url="https://api.deepseek.com",
        )

    async def stream_generate(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )

        for chunk in response:
            if chunk.choices[0].delta.content:
                yield f"data: {json.dumps({'text': chunk.choices[0].delta.content})}\n\n"

        yield "data: [DONE]\n\n"
