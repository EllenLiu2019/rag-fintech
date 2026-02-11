from typing import List
import asyncio

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from common import get_logger
from common.exceptions import ParsingError, DocumentNotFoundError, EvaluationNotFoundError, SubgraphNotFoundError
from rag.ingestion.pipeline import pipeline_runner
from rag.entity import DocumentType
from agent.claims_orchestrator import ClaimsOrchestrator
from agent.graph_state import HumanDecision
from agent.medical_agents import graph
from rag.persistence.persistent_service import PersistentService

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/claim",
    tags=["Claim"],
    responses={404: {"description": "Not found"}},
)


class UploadClaim(BaseModel):
    task_id: str
    doc_id: str
    filename: str
    message: str


class SubmitClaimRequest(BaseModel):
    doc_id: str


class HumanDecisionItem(BaseModel):
    icd_concept_code: str
    icd_concept_name: str
    tnm_stage: str


class ApproveClaimRequest(BaseModel):
    doc_id: str
    thread_ids: List[str]
    decisions: List[HumanDecisionItem]


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
):
    logger.info(f"Received claim material upload: {file.filename}")

    try:
        ingestion_job = await pipeline_runner(file, doc_type=DocumentType.CLAIM)

        response = UploadClaim(
            task_id=ingestion_job.job_id,
            doc_id=ingestion_job.doc_id,
            filename=file.filename,
            message=f"File '{file.filename}' accepted",
        )
        return JSONResponse(status_code=202, content=response.model_dump())

    except ParsingError:
        raise
    except Exception as e:
        logger.error(f"Failed to process claim upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"理赔材料处理失败: {str(e)}")


@router.post("/submit")
async def submit_claim(request: SubmitClaimRequest):
    """Phase 1: Start claim evaluation. Returns AI results for human review.

    Response includes pending_reviews with ICD/TNM candidates for each entity.
    Frontend should display these for human confirmation, then call /approve.
    """
    logger.info(f"Received claim processing request for doc_id: {request.doc_id}")
    try:
        orchestrator = ClaimsOrchestrator()
        result = await orchestrator.start_evaluation(request.doc_id)

        return JSONResponse(status_code=200, content=result)
    except (DocumentNotFoundError, HTTPException):
        raise
    except Exception as e:
        logger.error(f"Failed to start claim evaluation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"理赔评估启动失败: {str(e)}")


@router.post("/approve")
async def approve_claim(request: ApproveClaimRequest):
    """Phase 2: Complete claim evaluation with human-confirmed decisions.

    Accepts the reviewed ICD codes and TNM stages, resumes graph execution,
    then runs clause matching and eligibility reasoning to produce final decision.
    """
    logger.info(f"Received claim approval for doc_id: {request.doc_id}")
    try:
        if len(request.thread_ids) != len(request.decisions):
            raise HTTPException(
                status_code=400,
                detail=f"thread_ids count ({len(request.thread_ids)}) must match decisions count ({len(request.decisions)})",
            )

        decisions = [
            HumanDecision(
                icd_concept_code=d.icd_concept_code,
                icd_concept_name=d.icd_concept_name,
                tnm_stage=d.tnm_stage,
            )
            for d in request.decisions
        ]

        orchestrator = ClaimsOrchestrator()
        claim_decision = await orchestrator.complete_evaluation(
            doc_id=request.doc_id,
            thread_ids=request.thread_ids,
            decisions=decisions,
        )

        return JSONResponse(status_code=200, content=claim_decision.to_dict())
    except (DocumentNotFoundError, HTTPException):
        raise
    except Exception as e:
        logger.error(f"Failed to complete claim evaluation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"理赔审批失败: {str(e)}")


# ---------------------------------------------------------------------------
# Time Travel endpoints
# ---------------------------------------------------------------------------


@router.get("/{doc_id}/evaluations")
async def list_evaluations(doc_id: str):
    """List all evaluation attempts for a given doc_id (for history display)."""
    try:
        records = await asyncio.to_thread(PersistentService.list_evaluations, doc_id)
        results = []
        for r in records:
            row = r.to_dict()
            for key in ("created_at", "updated_at"):
                if row.get(key) is not None:
                    row[key] = row[key].isoformat()
            results.append(row)
        return JSONResponse(status_code=200, content=results)
    except Exception as e:
        logger.error(f"Failed to list evaluations for doc_id={doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/checkpoints/{thread_id}")
async def list_checkpoints(thread_id: str, limit: int = Query(default=20, ge=1, le=100)):
    """List checkpoint timeline for a given thread_id (lightweight metadata only)."""
    logger.info(f"Listing checkpoints for thread_id={thread_id}, limit={limit}")
    try:
        config: dict = {"configurable": {"thread_id": thread_id}}
        checkpoints = await graph.list_checkpoints(config, limit=limit)
        return JSONResponse(status_code=200, content=checkpoints)
    except Exception as e:
        logger.error(f"Failed to list checkpoints for thread_id={thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state/{thread_id}")
async def get_graph_state(
    thread_id: str,
    checkpoint_id: str = Query(default=None),
):
    """Get full graph state at a specific checkpoint (or latest if checkpoint_id omitted)."""
    logger.info(f"Getting checkpoint state for thread_id={thread_id}, checkpoint_id={checkpoint_id}")
    try:
        config: dict = {"configurable": {"thread_id": thread_id}}
        if checkpoint_id:
            config["configurable"]["checkpoint_id"] = checkpoint_id
        state = await graph.get_state(config)
        return JSONResponse(status_code=200, content=state)
    except Exception as e:
        logger.error(f"Failed to get state for thread_id={thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Subgraph Time Travel endpoints
# ---------------------------------------------------------------------------


@router.get("/subgraph-checkpoints/{thread_id}")
async def list_subgraph_checkpoints(
    thread_id: str,
    subgraph_name: str = Query(..., description="Name of the subgraph (e.g. encode_graph, stage_graph)"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """List checkpoint timeline for a subgraph within a given thread.

    Requires that subgraph configs were captured during the interrupt
    (persisted in claim_evaluations.subgraph_configs).
    """
    logger.info(f"Listing subgraph checkpoints for thread_id={thread_id}, subgraph_name={subgraph_name}, limit={limit}")
    try:
        subgraph_config = await asyncio.to_thread(PersistentService.get_subgraph_config, thread_id, subgraph_name)
        checkpoints = await graph.list_checkpoints(subgraph_config, limit=limit)
        return JSONResponse(status_code=200, content=checkpoints)
    except (EvaluationNotFoundError, SubgraphNotFoundError):
        raise
    except Exception as e:
        logger.error(
            f"Failed to list subgraph checkpoints for thread_id={thread_id}, subgraph={subgraph_name}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/subgraph-state/{thread_id}")
async def get_subgraph_state(
    thread_id: str,
    subgraph_name: str = Query(..., description="Name of the subgraph (e.g. encode_graph, stage_graph)"),
):
    """Get subgraph state using the config captured at interrupt time."""
    try:
        subgraph_config = await asyncio.to_thread(PersistentService.get_subgraph_config, thread_id, subgraph_name)
        state = await graph.get_state(subgraph_config)
        return JSONResponse(status_code=200, content=state)
    except (EvaluationNotFoundError, SubgraphNotFoundError):
        raise
    except Exception as e:
        logger.error(
            f"Failed to get subgraph state for thread_id={thread_id}, subgraph={subgraph_name}: {e}",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))
