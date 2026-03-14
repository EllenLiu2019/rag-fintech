from rag.llm.chat_model import chat_model
from typing import List, Optional, Dict, Any, AsyncIterator
import json
from common.log_utils import get_logger
from common import get_base_config, model_registry, prompt_manager

logger = get_logger(__name__)


class LLMService:

    def __init__(self, model: Dict[str, Any]):
        self.llm = chat_model[model["provider"]](
            model_name=model["model_name"],
            base_url=model["base_url"],
        )

    def _prepare_messages(
        self,
        question: str,
        context: List[Dict[str, Any]],
        relevant_foc: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **kwargs,
    ) -> List[Dict[str, str]]:
        system_prompt_name = "QA_system_prompt"
        user_prompt_name = "QA_user_prompt"

        if kwargs.get("is_eval", False):
            system_prompt_name = f"{system_prompt_name}_eval"

        conversation_history = conversation_history or []

        # format context
        context_parts = []
        if relevant_foc:
            for idx, chunk in enumerate(context):
                if chunk.get("clause_id", "-1") == "-1":
                    chunk_text = chunk.get("text", "")
                    context_parts.append(f"[{idx + 1}]\n{chunk_text}")

            context_str = "\n\n".join(context_parts)
            context_str = f"## 保单内容：\n\n{context_str}---\n\n{relevant_foc}"
        else:
            for idx, chunk in enumerate(context):
                chunk_text = chunk.get("text", "")
                context_parts.append(f"[{idx + 1}]\n{chunk_text}")

            context_str = "\n\n".join(context_parts)

        logger.debug(f"Context: {context_str}")

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
        user_content = prompt_manager.get(
            user_prompt_name,
            history_text=history_text,
            context_str=context_str,
            question=question,
        )

        messages = [
            {"role": "system", "content": prompt_manager.get(system_prompt_name)},
            {"role": "user", "content": user_content},
        ]
        return messages

    def answer_question(
        self,
        question: str,
        context: List[Dict[str, Any]],
        relevant_foc: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        messages = self._prepare_messages(question, context, relevant_foc, conversation_history, **kwargs)

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
        relevant_foc: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Stream generation of answer
        Yields: SSE event strings
        """
        messages = self._prepare_messages(question, context, relevant_foc, conversation_history, **kwargs)

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
    chat_config = get_base_config("chat", {})
    model_name = chat_config.get("llm_service", "qa_reasoner")
    model_config = model_registry.get_chat_model(model_name)
    llm_service = LLMService(model=model_config.to_dict())

    logger.info("Initialized LLMService singleton")
    return llm_service


llm_service = _create_llm_service()
