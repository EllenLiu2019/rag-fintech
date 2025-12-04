from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional, Literal

from common.log_utils import get_logger
from rag.retrieval.retriever import Retriever
from rag.generation.llm_service import LLMService
from rag.dependencies import get_retriever, get_llm_service

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Chat"],
    responses={404: {"description": "Not found"}},
)


class ChatRequest(BaseModel):
    """聊天请求模型"""

    query: str
    kb_id: str = "default_kb"
    conversation_history: List[dict] = []
    filters: Optional[dict] = {}
    stream: bool = True
    generation_config: Optional[dict] = {}
    mode: Literal["dense", "hybrid"] = "dense"


class ChatResponse(BaseModel):
    """聊天响应模型"""

    answer: str
    reasoning: Optional[str] = None
    sources: List[dict] = []
    tokens: Optional[int] = None
    mode: Literal["dense", "hybrid"] = "dense"


@router.post("/chat", response_model=ChatResponse)
async def chat_qa(
    request: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    llm_service: LLMService = Depends(get_llm_service),
):
    """
    Standard chat endpoint with dependency injection.

    Services are injected as singletons.
    """
    try:
        logger.info(f"Received chat request: query='{request.query}', kb_id='{request.kb_id}'")

        top_k = request.generation_config.get("top_k", 5) if request.generation_config else 5
        filters = request.filters or {}

        logger.info(f"Retrieving top {top_k} chunks with filters: {filters}")
        retrieved_chunks = retriever.search(
            query=request.query,
            kb_id=request.kb_id,
            top_k=top_k,
            filters=filters,
            mode=request.mode,
        )

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")

        if retrieved_chunks:
            generation_config = request.generation_config or {}
            temperature = generation_config.get("temperature", 0.7)
            max_tokens = generation_config.get("max_tokens")

            logger.info(f"Generating answer with temperature={temperature}, max_tokens={max_tokens}")
            llm_result = llm_service.answer_question(
                question=request.query,
                context=retrieved_chunks,
                conversation_history=request.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = llm_result.get("answer", "")
            reasoning = llm_result.get("reasoning")
            tokens = llm_result.get("tokens", 0)
        else:
            answer = "抱歉，没有找到相关的文档片段来回答您的问题。请尝试调整查询或过滤器。"
            reasoning = "检索结果为空"
            tokens = 0

        return ChatResponse(
            answer=answer,
            reasoning=reasoning,
            sources=retrieved_chunks,
            tokens=tokens,
            mode=request.mode,
        )

    except Exception as e:
        logger.error(f"Chat Q&A failed: {str(e)}")
        import traceback

        logger.error(f"   error stack:\n{traceback.format_exc()}")
        return ChatResponse(
            answer=f"抱歉，处理您的问题时发生错误: {str(e)}",
            reasoning="检索过程中发生异常",
            sources=[],
            tokens=0,
            mode=request.mode,
        )


@router.post("/chat/stream")
async def chat_qa_stream(
    request: ChatRequest,
    retriever: Retriever = Depends(get_retriever),
    llm_service: LLMService = Depends(get_llm_service),
):
    """
    Stream version of chat_qa endpoint.

    Returns Server-Sent Events (SSE) stream.
    """
    from fastapi.responses import StreamingResponse
    import json

    async def event_generator():
        try:
            logger.info(f"Received chat stream request: query='{request.query}', kb_id='{request.kb_id}'")

            # 1. Retrieval (using injected retriever)
            top_k = request.generation_config.get("top_k", 5) if request.generation_config else 5
            filters = request.filters or {}

            logger.info(f"Retrieving top {top_k} chunks with filters: {filters}")
            retrieved_chunks = retriever.search(
                query=request.query,
                kb_id=request.kb_id,
                top_k=top_k,
                filters=filters,
                mode=request.mode,
            )

            # 2. Send chunks
            if retrieved_chunks:
                yield f"data: {json.dumps({'type': 'chunks', 'data': retrieved_chunks}, ensure_ascii=False)}\n\n"
            else:
                # No chunks found
                yield f"data: {json.dumps({'type': 'error', 'data': {'message': '抱歉，没有找到相关的文档片段来回答您的问题。'}}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 3. Stream generation (using injected llm_service)
            generation_config = request.generation_config or {}
            temperature = generation_config.get("temperature", 0.7)
            max_tokens = generation_config.get("max_tokens")

            async for event in llm_service.stream_answer_question(
                question=request.query,
                context=retrieved_chunks,
                conversation_history=request.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield event

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Chat stream failed: {str(e)}")
            import traceback

            logger.error(f"   error stack:\n{traceback.format_exc()}")
            # This enables retry mechanism in the frontend
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': str(e)}}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
