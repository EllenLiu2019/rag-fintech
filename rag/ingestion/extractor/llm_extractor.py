import json
import re
from typing import Any
from rag.llm.chat_model import chat_model
from common import prompt_manager


class LLMExtractor:

    def __init__(self, model: dict[str, Any]):
        self.llm = chat_model[model["provider"]](
            model_name=model["model_name"],
            base_url=model["base_url"],
        )

    def _build_prompt(self, content: str, hints: dict = None, missing_fields: dict = None) -> str:
        fields = missing_fields or {}
        return prompt_manager.get(
            "insurance_extraction",
            hints=json.dumps(hints or {}, ensure_ascii=False),
            content=content,
            fields=json.dumps(fields, ensure_ascii=False),
        )

    @staticmethod
    def _parse_response(content: str) -> dict:
        cleaned = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)

    def extract(self, content: str, hints: dict = None, missing_fields: dict = None) -> dict:
        prompt = self._build_prompt(content, hints, missing_fields)
        _, content, tokens = self.llm.generate(prompt=prompt, temperature=0)
        return {
            "content": self._parse_response(content),
            "tokens": tokens,
        }

    async def aextract(self, content: str, hints: dict = None, missing_fields: dict = None) -> dict:
        """Async version using LLM's native agenerate()."""
        prompt = self._build_prompt(content, hints, missing_fields)
        _, content, tokens = await self.llm.agenerate(prompt=prompt, temperature=0)
        return {
            "content": self._parse_response(content),
            "tokens": tokens,
        }
