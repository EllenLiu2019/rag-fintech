import json
import re
from typing import Dict, Any, List, Literal

from rag.llm.chat_model import chat_model
from common.log_utils import get_logger
from common.prompt_manager import get_prompt_manager

logger = get_logger(__name__)


class BaseRewriter:
    def __init__(self, model: dict[str, Any], temperature: float = 0.0):
        self.llm = chat_model[model["provider"]](model_name=model["model_name"])
        self.temperature = temperature
        self.prompt_manager = get_prompt_manager()

    def _build_prompt(self) -> str:
        pass

    def _clean_response(self, content: str) -> str:
        if not content:
            return ""

        result = content.strip()

        # Remove quotes if present
        if (result.startswith('"') and result.endswith('"')) or (result.startswith("'") and result.endswith("'")):
            result = result[1:-1]

        # Remove markdown code blocks if present
        if result.startswith("```") and result.endswith("```"):
            result = re.sub(r"```(?:\w+)?\n?", "", result).strip()

        # Remove "输出:" prefix if present
        if result.startswith("输出:") or result.startswith("输出："):
            result = result[3:].strip()

        return result


class UnifiedRewriter(BaseRewriter):

    def __init__(
        self,
        model: dict[str, Any],
        temperature: float = 0.0,
        history_max_length: int = 10,
    ):
        super().__init__(model=model, temperature=temperature)
        self.history_max_length = history_max_length
        self.histories: List[str] = []

    def _build_prompt(self) -> str:
        history_str = json.dumps(self.histories[-self.history_max_length :], ensure_ascii=False)
        return self.prompt_manager.get("unified_rewrite", histories=history_str)

    def rewrite(self, query: str) -> Dict[str, Any]:
        try:
            reasoning, content, tokens = self.llm.generate(
                messages=[
                    {"role": "system", "content": self._build_prompt()},
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
            )

            rewritten = self._clean_response(content)
            self.histories.append(query)

            logger.info(f"Query rewritten: '{query}' -> '{rewritten}'")
            return {
                "rewritten_query": rewritten,
                "tokens": tokens,
            }

        except Exception as e:
            logger.error(f"Query rewrite failed: {e}", exc_info=True)
            return {
                "rewritten_query": query,
                "tokens": 0,
            }

    def clear_history(self) -> None:
        self.histories.clear()

    def add_to_history(self, query: str) -> None:
        self.histories.append(query)


class HyDERewriter(BaseRewriter):
    def __init__(self, model: dict[str, Any], temperature: float = 0.3):
        super().__init__(model=model, temperature=temperature)

    def _build_prompt(self) -> str:
        return self.prompt_manager.get("hyde_rewrite")

    def rewrite(self, query: str) -> Dict[str, Any]:
        try:
            reasoning, content, tokens = self.llm.generate(
                messages=[
                    {"role": "system", "content": self._build_prompt()},
                    {"role": "user", "content": query},
                ],
                temperature=self.temperature,
            )

            hypothetical_doc = content.strip() if content else query
            logger.info(f"HyDE generated: '{query}' -> '{hypothetical_doc}...'")

            return {
                "rewritten_query": hypothetical_doc,
                "tokens": tokens,
            }

        except Exception as e:
            logger.error(f"HyDE generation failed: {e}", exc_info=True)
            return {
                "rewritten_query": query,
                "tokens": 0,
            }


class GlossaryInjector:

    def __init__(self):
        self.glossary = {
            "保险费": "保费",
            "保险金": "保额",
            "赔钱": "理赔",
            "交钱": "缴费",
            "买保险": "投保",
            "退保险": "退保",
        }

    def inject(self, query: str) -> str:
        result = query
        for colloquial, professional in self.glossary.items():
            result = result.replace(colloquial, professional)
        return result


class QueryOptimizer:

    def __init__(self, model: dict[str, Any]):
        self.model = model
        self.unified_rewriter = UnifiedRewriter(model=model)
        self.hyde_rewriter = HyDERewriter(model=model)
        self.glossary_injector = GlossaryInjector()

    def optimize(self, query: str, mode: Literal["unified", "hyde"] = "unified") -> Dict[str, Any]:
        if mode == "unified":
            rewrite_result = self.unified_rewriter.rewrite(query)
            rewritten_query = rewrite_result["rewritten_query"]
            optimized_query = self.glossary_injector.inject(rewritten_query)
        elif mode == "hyde":
            rewrite_result = self.hyde_rewriter.rewrite(query)
            optimized_query = rewrite_result["rewritten_query"]
        else:
            raise ValueError(f"Invalid optimization mode: {mode}")

        return {
            "optimized_query": optimized_query,
            "tokens": rewrite_result.get("tokens", 0),
        }

    def clear_history(self) -> None:
        self.unified_rewriter.clear_history()

    def add_to_history(self, query: str) -> None:
        self.unified_rewriter.add_to_history(query)
