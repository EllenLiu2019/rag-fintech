from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from common import get_logger
from common.exceptions import ParsingError
from rag.ingestion.pipeline import pipeline_runner
from rag.entity import DocumentType

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


@router.post("/process")
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
