import uuid
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from enum import Enum

from common import get_logger
from rag.entity.clause_tree import ClauseForest

logger = get_logger(__name__)


class DocumentType(Enum):
    POLICY = "policy"
    CLAIM = "claim"
    MEDICAL_RECORD = "medical"


class RagDocument(BaseModel):

    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="document_id")
    job_id: str = Field(default="", description="job_id")
    pages: list[dict[str, Any]] = Field(default_factory=list, description="pages")
    business_data: dict[str, Any] = Field(default_factory=dict, description="business_data")
    confidence: dict[str, Any] = Field(default_factory=dict, description="extracted_data confidence")
    token_num: int = 0
    chunk_num: int = 0
    filename: str
    file_size: int = 0
    content_type: Optional[str] = None
    page_count: int = 0
    upload_time: Optional[str] = None
    clause_forest: Optional[ClauseForest] = Field(default=None, description="clause forest structure")
    doc_type: DocumentType = Field(default=DocumentType.POLICY, description="document type")
    chunks: list[dict[str, Any]] = Field(default_factory=list, description="chunks")

    @field_validator("pages", mode="before")
    @classmethod
    def validate_pages(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError(f"Expected list for pages, got {type(v).__name__}")
        return v

    @field_validator("confidence", "business_data", mode="before")
    @classmethod
    def validate_dict_fields(cls, v):
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError(f"Expected dict, got {type(v).__name__}")
        return v

    @classmethod
    def from_extraction_result(
        cls,
        document_id: str,
        parsed_documents: list[dict[str, Any]],
        business_data: dict[str, Any] = {},
        confidence: dict[str, Any] = {},
        token_num: int = 0,
        filename: str = "",
        file_size: int = 0,
        content_type: Optional[str] = None,
        chunk_num: int = 0,
        job_id: str = "",
        clause_forest: Optional[ClauseForest] = None,
        doc_type: DocumentType = DocumentType.POLICY,
    ) -> "RagDocument":
        return cls(
            document_id=document_id,
            pages=parsed_documents,
            business_data=business_data,
            confidence=confidence,
            token_num=token_num,
            chunk_num=chunk_num,
            filename=filename,
            file_size=file_size,
            content_type=content_type,
            page_count=len(parsed_documents),
            upload_time=datetime.now(tz=timezone.utc).isoformat(),
            job_id=job_id,
            clause_forest=clause_forest,
            doc_type=doc_type,
        )

    def to_document_metadata(self) -> dict[str, Any]:
        """
        convert to document metadata dictionary (for vector database)

        only contains information for retrieval

        Returns:
            dict[str, Any]: document metadata dictionary
        """
        return {
            "document_id": self.document_id,
            "doc_type": self.doc_type.value,
            "job_id": self.job_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "content_type": self.content_type,
            "page_count": self.page_count,
            "upload_time": self.upload_time,
        }

    def to_parsed_file(self) -> dict[str, Any]:

        return {
            **self.to_document_metadata(),
            "pages": self.pages,
            "business_data": self.business_data,
            "overall_confidence": self.confidence.get("overall_confidence"),
        }

    def is_complete(self) -> bool:

        return bool(self.filename) and bool(self.business_data)
