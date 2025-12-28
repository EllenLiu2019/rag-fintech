class RagBaseException(Exception):
    """Base class for all RAG exceptions"""

    def __init__(self, message: str, code: str = None, details: dict = None):
        self.message = message
        self.code = code or self.__class__.__name__
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict:
        return {"error_code": self.code, "message": self.message, "details": self.details}


# ============ API Layer Exceptions ============


class APIError(RagBaseException):
    """API layer generic exception"""

    http_status = 500


class ValidationError(APIError):
    """Request parameter validation failed"""

    http_status = 400


class AuthenticationError(APIError):
    """Authentication failed"""

    http_status = 401


class NotFoundError(APIError):
    """Resource not found"""

    http_status = 404


class RateLimitExceededError(APIError):
    """Request frequency limit exceeded"""

    http_status = 429


# ============ Service Layer Exceptions ============


class ServiceError(RagBaseException):
    """Service layer generic exception"""

    pass


class IngestionError(ServiceError):
    """Document ingestion failed"""

    pass


class ParsingError(IngestionError):
    """Document parsing failed"""

    pass


class ExtractionError(IngestionError):
    """Information extraction failed"""

    pass


class ChunkingError(IngestionError):
    """Document chunking failed"""

    pass


class RetrievalError(ServiceError):
    """Retrieval failed"""

    pass


class RerankError(ServiceError):
    """Reranking failed"""

    pass


class GenerationError(ServiceError):
    """Generation failed"""

    pass


# ============ Integration Layer Exceptions ============


class LLMError(RagBaseException):
    """LLM call exception base class"""

    pass


class ModelNotFoundError(LLMError):
    """Model not found"""

    pass


class ModelTimeoutError(LLMError):
    """Model call timeout"""

    pass


class ModelServerError(LLMError):
    """Model API server error (5xx)"""

    def __init__(self, message: str, status_code: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.status_code = status_code


class ModelRateLimitError(LLMError):
    """Model API rate limit exceeded"""

    def __init__(self, message: str, retry_after: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class EmbeddingError(LLMError):
    """Embedding generation failed"""

    pass


class TokenLimitExceededError(LLMError):
    """Token limit exceeded"""

    def __init__(self, message: str, token_count: int = None, limit: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.token_count = token_count
        self.limit = limit


# ============ Repository Layer Exceptions ============


class StorageError(RagBaseException):
    """Storage layer generic exception"""

    pass


class ConnectionError(StorageError):
    """Connection failed"""

    pass


class VectorStoreError(StorageError):
    """Vector database exception"""

    pass


class DatabaseError(StorageError):
    """Relational database exception"""

    pass


class CacheError(StorageError):
    """Cache exception"""

    pass


class FileStorageError(StorageError):
    """File storage exception"""

    pass


class DocumentNotFoundError(DatabaseError):
    """Document not found"""

    pass
