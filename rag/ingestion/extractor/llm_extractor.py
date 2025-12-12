import json
from typing import Any
from rag.llm.chat_model import llm_model
from common.prompt_manager import get_prompt_manager


class LLMExtractor:

    def __init__(self, model: dict[str, Any]):
        self.llm = llm_model[model["provider"]](model_name=model["model_name"])
        self.prompt_manager = get_prompt_manager()

    def extract(self, content: str, hints: dict = None) -> dict:

        fields = {
            "total_premium": {"总保费": ""},
            "coverage_amount": {"保险金额": ""},
            "confidence": 0.95,
        }

        prompt = self.prompt_manager.get(
            "insurance_extraction",
            hints=json.dumps(hints or {}, ensure_ascii=False),
            content=content,
            fields=json.dumps(fields, ensure_ascii=False),
        )

        _, content, tokens = self.llm.generate(prompt=prompt, temperature=0)
        content = json.loads(content.replace("```json", "").replace("```", ""))
        return {
            "content": content,
            "tokens": tokens,
        }
