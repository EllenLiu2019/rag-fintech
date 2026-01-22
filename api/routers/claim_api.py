from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from common import get_logger
from common.exceptions import ParsingError, DocumentNotFoundError
from rag.ingestion.pipeline import pipeline_runner
from rag.entity import DocumentType
from agent.claims_orchestrator import ClaimsOrchestrator

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
async def submit_claim(
    request: SubmitClaimRequest,
):
    logger.info(f"Received claim processing request for doc_id: {request.doc_id}")
    try:
        orchestrator = ClaimsOrchestrator()
        claim_decision = await orchestrator.evaluate_claim(request.doc_id)

        return JSONResponse(status_code=200, content=claim_decision.to_dict())
    except (DocumentNotFoundError, HTTPException):
        raise
    except Exception as e:
        logger.error(f"Failed to process claim: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"理赔处理失败: {str(e)}")
