from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Literal

from common import get_logger
from rag.retrieval import retriever

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
    foc_enhance: bool = True


@router.post("/search")
async def search_docs(request: SearchRequest):
    """
    Search documents using vector retrieval.

    - **query**: Search query string
    - **kb_id**: Knowledge base ID
    - **top_k**: Number of results to return
    - **filters**: Metadata filters
    - **mode**: Retrieval mode (dense or hybrid)
    - **foc_enhance**: Whether to enhance the results with clause forest
    """
    logger.info(f"Received search request: query='{request.query}', kb_id='{request.kb_id}', top_k={request.top_k}")

    # Let RetrievalError propagate to global handler
    results = retriever.search(
        query=request.query,
        kb_id=request.kb_id,
        top_k=request.top_k,
        filters=request.filters,
        mode=request.mode,
        foc_enhance=request.foc_enhance,
    )

    formatted_results = results or []

    return JSONResponse(
        status_code=200,
        content={
            "query": request.query,
            "results": formatted_results["results"],
            "foc_markdown": formatted_results["foc_markdown"],
            "foc_data": formatted_results.get("foc_data"),
            "query_to_use": formatted_results["query_to_use"],
            "snomed_entities": formatted_results["snomed_entities"],
            "total": len(formatted_results["results"]),
            "mode": request.mode,
        },
    )
