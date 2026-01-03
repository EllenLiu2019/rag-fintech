from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
from llama_index.core.schema import Document as LlamaIndexDocument
from common import get_logger

logger = get_logger(__name__)


class ApiParsedDocument(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        exclude_none=False,
        validate_assignment=True,
    )

    text: str = Field(default="", description="document text content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="document metadata")
    id: Optional[str] = Field(default=None, description="document unique identifier")

    @field_validator("text", mode="before")
    @classmethod
    def validate_text(cls, v):
        if v is None:
            return ""
        return str(v)

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, v):
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError(f"metadata must be a dict, got {type(v).__name__}")
        return v

    @classmethod
    def from_document(cls, doc: LlamaIndexDocument) -> "ApiParsedDocument":
        if not isinstance(doc, LlamaIndexDocument):
            raise TypeError(f"Expected LlamaIndexDocument, got {type(doc).__name__}")

        # extract id
        doc_id = None
        if hasattr(doc, "id_"):
            try:
                doc_id = str(doc.id_) if doc.id_ else None
            except (AttributeError, TypeError):
                doc_id = None

        return cls(text=doc.text or "", metadata=doc.metadata or {}, id=doc_id)

    def to_document(self) -> LlamaIndexDocument:
        doc = LlamaIndexDocument(text=self.text, metadata=self.metadata)
        if self.id:
            doc.id_ = self.id
        return doc

    def to_dict(self, exclude_none: bool = False) -> Dict[str, Any]:
        return self.model_dump(exclude_none=exclude_none)

    def to_json(
        self,
        exclude_none: bool = False,
        indent: Optional[int] = None,
        ensure_ascii: bool = False,
    ) -> str:
        return self.model_dump_json(exclude_none=exclude_none, indent=indent, ensure_ascii=ensure_ascii)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ApiParsedDocument":
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")

        try:
            # Handle id_ to id conversion for backward compatibility
            if "id_" in data and "id" not in data:
                data = data.copy()
                data["id"] = data["id_"]
            return cls(**data)
        except Exception as e:
            logger.error(f"Failed to deserialize ApiParsedDocument from dict: {data}, error: {e}")
            raise ValueError(f"Invalid document data: {e}") from e


def to_dict(doc: LlamaIndexDocument) -> Dict[str, Any]:
    model = ApiParsedDocument.from_document(doc)
    return model.model_dump(exclude_none=False)


def from_dict(data: Dict[str, Any]) -> LlamaIndexDocument:
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")

    try:
        # Handle id_ to id conversion for backward compatibility
        if "id_" in data and "id" not in data:
            data = data.copy()
            data["id"] = data["id_"]

        model = ApiParsedDocument.from_dict(data)
        return model.to_document()
    except Exception as e:
        logger.error(f"Failed to deserialize document from dict: {data}, error: {e}")
        raise ValueError(f"Invalid document data: {e}") from e


def _register_serializers():
    """
    Register LlamaIndexDocument serializers with JSONMarshaller.
    This enables json_marshaller to correctly serialize/deserialize LlamaIndexDocument objects.
    """
    try:
        from rag.marshaller.json_marshaller import JSONMarshaller

        JSONMarshaller.register_serializer(LlamaIndexDocument, to_dict)
        JSONMarshaller.register_deserializer(LlamaIndexDocument, from_dict)
        logger.debug("Registered LlamaIndexDocument serializers with JSONMarshaller")
    except ImportError:
        logger.debug("JSONMarshaller not available, skipping serializer registration")
    except Exception as e:
        logger.warning(f"Failed to register serializers: {e}")


# Auto-register serializers on module import
_register_serializers()
