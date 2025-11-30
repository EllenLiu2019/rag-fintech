from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from common.log_utils import get_logger
from service.retrieval.retriever import Retriever
from service.llm_service import LLMService

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


class ChatResponse(BaseModel):
    """聊天响应模型"""

    answer: str
    reasoning: Optional[str] = None
    sources: List[dict] = []
    tokens: Optional[int] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_qa(request: ChatRequest):

    try:
        logger.info(f"Received chat request: query='{request.query}', kb_id='{request.kb_id}'")

        retriever = Retriever()
        top_k = request.generation_config.get("top_k", 5) if request.generation_config else 5
        filters = request.filters or {}

        logger.info(f"Retrieving top {top_k} chunks with filters: {filters}")
        retrieved_chunks = retriever.search(
            query=request.query,
            kb_id=request.kb_id,
            top_k=top_k,
            filters=filters,
        )

        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")

        if retrieved_chunks:
            llm_service = LLMService()

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
        )


@router.post("/chat/stream")
async def chat_qa_stream(request: ChatRequest):
    """
    Stream version of chat_qa endpoint

    Returns Server-Sent Events (SSE) stream
    """
    # TODO: Phase 2
    from fastapi.responses import StreamingResponse
    import json

    async def event_generator():
        # 模拟流式输出
        yield f"data: {json.dumps({'type': 'chunks', 'data': []})}\n\n"
        yield f"data: {json.dumps({'type': 'reasoning', 'data': 'Placeholder reasoning'})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'data': 'This '})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'data': 'is '})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'data': 'a '})}\n\n"
        yield f"data: {json.dumps({'type': 'token', 'data': 'placeholder'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
