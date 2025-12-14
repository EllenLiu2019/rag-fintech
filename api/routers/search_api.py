from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Literal

from common import get_logger
from rag.retrieval.retriever import retriever

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api",
    tags=["Search"],
    responses={404: {"description": "Not found"}},
)


class SearchRequest(BaseModel):
    query: str
    kb_id: str = "default_kb"
    top_k: int = 5
    filters: dict
    mode: Literal["dense", "hybrid"] = "dense"


@router.post("/search")
async def search_docs(request: SearchRequest):
    try:
        results = retriever.search(
            query=request.query,
            kb_id=request.kb_id,
            top_k=request.top_k,
            filters=request.filters,
            mode=request.mode,
        )

        formatted_results = results or []

        return JSONResponse(
            status_code=200,
            content={
                "query": request.query,
                "results": formatted_results,
                "total": len(formatted_results),
                "mode": request.mode,
            },
        )
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        import traceback

        logger.error(f"   error stack:\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"message": f"搜索失败: {str(e)}"})
