from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from common.log_utils import get_logger
from service.retrieval.retriever import Retriever

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


@router.post("/search")
async def search_docs(request: SearchRequest):

    retriever = Retriever()
    try:
        results = retriever.search(
            query=request.query, kb_id=request.kb_id, top_k=request.top_k, filters=request.filters
        )

        formatted_results = results or []

        return JSONResponse(
            status_code=200,
            content={"query": request.query, "results": formatted_results, "total": len(formatted_results)},
        )
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        import traceback

        logger.error(f"   error stack:\n{traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"message": f"搜索失败: {str(e)}"})
