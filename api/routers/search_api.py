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
    """
    Search documents using vector retrieval.

    - **query**: Search query string
    - **kb_id**: Knowledge base ID
    - **top_k**: Number of results to return
    - **filters**: Metadata filters
    - **mode**: Retrieval mode (dense or hybrid)
    """
    logger.info(f"Received search request: query='{request.query}', kb_id='{request.kb_id}', top_k={request.top_k}")

    # Let RetrievalError propagate to global handler
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
            "success": True,
            "data": {
                "query": request.query,
                "results": formatted_results,
                "total": len(formatted_results),
                "mode": request.mode,
            },
        },
    )
