from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal, List, Dict, Any
from typing import Optional

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
    foc_enhance: Optional[bool] = None


class SearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    relevant_foc: Optional[str] = None
    snomed_entities: Optional[dict] = None
    foc_data: Optional[dict] = None


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

    if not request.query or not request.query.strip():
        logger.warning("Empty query provided, returning empty results")
        return {}

    logger.info(
        f"Received search request: query='{request.query}', kb_id='{request.kb_id}', top_k={request.top_k}, mode={request.mode}, foc_enhance={request.foc_enhance}"
    )

    # Let RetrievalError propagate to global handler
    results = await retriever.search(
        query=request.query,
        kb_id=request.kb_id,
        top_k=request.top_k,
        filters=request.filters,
        mode=request.mode,
        foc_enhance=request.foc_enhance,
    )

    formatted_results = results or {
        "results": [],
        "relevant_foc": None,
        "snomed_entities": {},
        "foc_data": None,
    }

    return SearchResponse(
        query=request.query,
        results=formatted_results["results"],
        relevant_foc=formatted_results["relevant_foc"],
        snomed_entities=formatted_results["snomed_entities"],
        foc_data=formatted_results["foc_data"],
    )
