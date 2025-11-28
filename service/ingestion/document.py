import logging
from typing import Any, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

logger = logging.getLogger(__name__)


class RagDocument(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, exclude_none=False, validate_assignment=True)

    document_id: Optional[str] = Field(default=None, description="document_id")
    filename: str = Field(description="filename")
    file_size: int = Field(default=0, description="file_size")
    content_type: Optional[str] = Field(default=None, description="content_type")
    text: str = Field(default="", description="text")
    pages: list[dict[str, Any]] = Field(default_factory=list, description="pages")
    confidence: dict[str, Any] = Field(default_factory=dict, description="extracted_data confidence")
    metadata: dict[str, Any] = Field(default_factory=dict, description="metadata")

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, v):
        if v is None:
            return ""
        return str(v)

    @field_validator("pages", mode="before")
    @classmethod
    def validate_pages(cls, v):
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError(f"Expected list for pages, got {type(v).__name__}")
        return v

    @field_validator("confidence", "metadata", mode="before")
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
        parsed_documents: list[dict[str, Any]],
        confidence: dict[str, Any],
        metadata: dict[str, Any],
        filename: str,
        file_size: int = 0,
        content_type: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> "RagDocument":
        return cls(
            document_id=document_id,
            filename=filename,
            file_size=file_size,
            content_type=content_type,
            pages=parsed_documents,
            text="\n\n".join([doc["text"] or "" for doc in parsed_documents]),
            confidence=confidence,
            metadata=metadata,
        )

    def to_storage_dict(self) -> dict[str, Any]:
        """
        convert to storage dictionary (for storage)

        contains all necessary information, for subsequent retrieval and reconstruction

        Returns:
            dict[str, Any]: storage dictionary
        """
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "content_type": self.content_type,
            "text": self.text,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "pages": self.pages,
        }

    def to_index_dict(self) -> dict[str, Any]:
        """
        convert to index dictionary (for vector database)

        only contains information for retrieval

        Returns:
            dict[str, Any]: index dictionary
        """
        return {
            "id": self.document_id or f"{self.filename}_{hash(self.text)}",
            "text": self.text,
            "metadata": {
                **self.metadata,
                "filename": self.filename,
                "file_size": self.file_size,
                "content_type": self.content_type,
            },
        }

    def get_summary(self) -> dict[str, Any]:

        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_size": self.file_size,
            "text_length": len(self.text),
            "page_count": len(self.pages),
            "metadata_keys": list(self.metadata.keys()),
            "metadata": self.metadata,  # Add full metadata for frontend use
            "overall_confidence": self.confidence.get("overall_confidence"),
        }

    def is_complete(self) -> bool:

        return bool(self.text) and bool(self.filename) and bool(self.metadata)
