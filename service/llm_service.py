from rag.llm.chat_model import DeepSeek
from typing import List, Optional, Dict, Any
from common.log_utils import get_logger

logger = get_logger(__name__)


class LLMService:

    def __init__(self):
        self.llm = DeepSeek()

    def answer_question(
        self,
        question: str,
        context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:

        conversation_history = conversation_history or []

        # format context, add reference annotation
        context_parts = []
        for idx, chunk in enumerate(context):
            chunk_text = chunk.get("text", "")
            chunk_score = chunk.get("score", 0.0)
            context_parts.append(f"[{idx + 1}] (相似度: {chunk_score:.2f})\n{chunk_text}")

        context_str = "\n\n".join(context_parts)

        # build system prompt
        system_prompt = """你是一个专业的保险领域智能问答助手。请遵循以下规则：

        1. **基于上下文回答**：严格根据提供的上下文信息回答问题，不要使用先验知识
        2. **明确投被保人信息**：根据保险合同信息明确投保人、被保险人信息
        3. **验证投被保人**：验证问题中指代的投保人和被保险人是否在合同中存在，如果不存在或不正确，请明确说明无法验证
        4. **引用来源**：在回答中使用 [1][2] 等标注引用来源（对应上下文中的编号）
        5. **简洁准确**：只回答问题中提到的内容，不要添加无关信息
        6. **多轮对话**：如果对话历史中有相关信息，可以结合历史上下文和当前检索结果回答
        """

        user_content_parts = [
            f"### 参考信息\n{context_str}\n",
            f"### 问题\n{question}\n",
            "### 回答要求\n请根据参考信息回答问题，并在答案中使用 [1][2] 等标注引用来源。",
        ]
        user_content = "\n".join(user_content_parts)

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            for msg in conversation_history:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": msg.get("content", "")})

        messages.append({"role": "user", "content": user_content})

        logger.info(f"Generating answer with temperature={temperature}, max_tokens={max_tokens}")

        reasoning, content, tokens = self.llm.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return {
            "answer": content,
            "reasoning": reasoning,
            "tokens": tokens,
        }
