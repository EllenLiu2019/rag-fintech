from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional, Literal

from common import get_logger
from rag.retrieval import retriever
from rag.generation import llm_service

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
    foc_enhance: bool = True


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
):
    """
    Standard chat endpoint with dependency injection.
    """
    if not request.query or not request.query.strip():
        logger.warning("Empty query provided, returning empty results")
        return {}

    logger.info(f"Received chat request: query='{request.query}', kb_id='{request.kb_id}'")

    top_k = request.generation_config.get("top_k", 5) if request.generation_config else 5
    filters = request.filters or {}

    logger.info(f"Retrieving top {top_k} chunks with filters: {filters}")
    # Let RetrievalError propagate
    retrieved_res = await retriever.search(
        query=request.query,
        kb_id=request.kb_id,
        top_k=top_k,
        filters=filters,
        mode=request.mode,
        foc_enhance=request.foc_enhance,
    )

    logger.info(f"Retrieved {len(retrieved_res['results'])} chunks")

    if retrieved_res["results"]:
        generation_config = request.generation_config or {}
        temperature = generation_config.get("temperature", 0.7)
        max_tokens = generation_config.get("max_tokens")

        logger.info(f"Generating answer with temperature={temperature}, max_tokens={max_tokens}")
        # Let GenerationError propagate
        llm_result = llm_service.answer_question(
            question=retrieved_res["query_to_use"],
            context=retrieved_res["results"],
            relevant_foc=retrieved_res.get("relevant_foc", None),
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
        sources=retrieved_res["results"],
        tokens=tokens,
        mode=request.mode,
    )


@router.post("/chat/stream")
async def chat_qa_stream(
    request: ChatRequest,
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
            retrival_res = await retriever.search(
                query=request.query,
                kb_id=request.kb_id,
                top_k=top_k,
                filters=filters,
                mode=request.mode,
                foc_enhance=request.foc_enhance,
            )

            # 2. Send chunks
            if retrival_res["results"]:
                yield f"data: {json.dumps({'type': 'chunks', 'data': retrival_res['results']}, ensure_ascii=False)}\n\n"
            else:
                # No chunks found
                yield f"data: {json.dumps({'type': 'error', 'data': {'message': '抱歉，没有找到相关的文档片段来回答您的问题。'}}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 3. Stream generation (using injected llm_service)
            generation_config = request.generation_config or {}
            temperature = generation_config.get("temperature", 1.0)
            max_tokens = generation_config.get("max_tokens")

            async for event in llm_service.stream_answer_question(
                question=retrival_res["query_to_use"],
                context=retrival_res["results"],
                relevant_foc=retrival_res.get("relevant_foc", None),
                conversation_history=request.conversation_history,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                yield event

            yield "data: [DONE]\n\n"

        except Exception as e:
            # Log error (will be handled by global handler for non-streaming)
            logger.error(f"Chat stream failed: {str(e)}", exc_info=True)
            # Send error to frontend for retry mechanism
            error_message = str(e)
            if hasattr(e, "code"):
                error_message = f"[{e.code}] {e.message}"
            yield f"data: {json.dumps({'type': 'error', 'data': {'message': error_message}}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
