from rag.llm.chat_model import chat_model
from typing import List, Optional, Dict, Any, AsyncIterator
import json
from common.log_utils import get_logger
from common import get_model_registry
from common.prompt_manager import get_prompt_manager

logger = get_logger(__name__)


class LLMService:

    def __init__(self, model: Dict[str, Any]):
        self.llm = chat_model[model["provider"]](
            model_name=model["model_name"],
            base_url=model["base_url"],
        )
        self.prompt_manager = get_prompt_manager()

    def _prepare_messages(
        self,
        question: str,
        context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        conversation_history = conversation_history or []

        # format context, add reference annotation
        context_parts = []
        for idx, chunk in enumerate(context):
            chunk_text = chunk.get("text", "")
            chunk_score = chunk.get("score", 0.0)
            # add reference annotation and similarity score
            context_parts.append(f"[{idx + 1}] (相似度: {chunk_score:.2f})\n{chunk_text}")

        context_str = "\n\n".join(context_parts)

        # format conversation history
        history_text = ""
        if conversation_history:
            history_parts = []
            for i in range(0, len(conversation_history), 2):
                q = conversation_history[i].get("content", "")
                a = conversation_history[i + 1].get("content", "") if i + 1 < len(conversation_history) else ""
                history_parts.append(f"Q: {q}\nA: {a}")
            history_text = "\n\n".join(history_parts)

        # user message with context and current question
        user_content = self.prompt_manager.get(
            "QA_user_prompt",
            history_text=history_text,
            context_str=context_str,
            question=question,
        )

        messages = [
            {"role": "system", "content": self.prompt_manager.get("QA_system_prompt")},
            {"role": "user", "content": user_content},
        ]
        return messages

    def answer_question(
        self,
        question: str,
        context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:

        messages = self._prepare_messages(question, context, conversation_history)

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

    async def stream_answer_question(
        self,
        question: str,
        context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """
        Stream generation of answer
        Yields: SSE event strings
        """
        messages = self._prepare_messages(question, context, conversation_history)

        logger.info(f"Streaming answer with temperature={temperature}, max_tokens={max_tokens}")

        reasoning_buffer = ""
        answer_buffer = ""
        total_tokens = 0

        async for chunk in self.llm.stream_generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            if chunk["type"] == "reasoning":
                reasoning_buffer += chunk["content"]
                yield f"data: {json.dumps({'type': 'reasoning', 'data': chunk['content'], 'done': False}, ensure_ascii=False)}\n\n"
            elif chunk["type"] == "content":
                answer_buffer += chunk["content"]
                yield f"data: {json.dumps({'type': 'token', 'data': chunk['content']}, ensure_ascii=False)}\n\n"
            elif chunk["type"] == "metadata":
                total_tokens = chunk.get("tokens", 0)
                logger.info(f"Stream completed with {total_tokens} total tokens")

        # Send complete reasoning signal
        if reasoning_buffer:
            yield f"data: {json.dumps({'type': 'reasoning', 'data': '', 'done': True}, ensure_ascii=False)}\n\n"

        # Send done signal with tokens information
        yield f"data: {json.dumps({'type': 'done', 'data': {'answer': answer_buffer, 'tokens': total_tokens}}, ensure_ascii=False)}\n\n"


def _create_llm_service() -> LLMService:
    """
    Create LLMService instance at module load time.
    """
    registry = get_model_registry()
    model_config = registry.get_chat_model("qa_reasoner")
    llm_service = LLMService(model=model_config.to_dict())

    logger.info("Initialized LLMService singleton")
    return llm_service


llm_service = _create_llm_service()
