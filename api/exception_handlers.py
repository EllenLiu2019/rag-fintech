from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from common.exceptions import RagBaseException, APIError, ValidationError
from common.error_codes import ErrorCodes
from common import get_logger

logger = get_logger(__name__)


async def rag_exception_handler(request: Request, exc: RagBaseException):
    """Handle all RAG custom exceptions"""

    # Determine HTTP status code
    if isinstance(exc, APIError):
        status_code = exc.http_status
    else:
        # Service/LLM/Repository layer exceptions -> 500
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR

    # Log error with context
    logger.error(
        f"[{exc.code}] {exc.message}",
        extra={
            "error_code": exc.code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": exc.to_dict()},
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""

    errors = exc.errors()
    error_messages = [f"{err['loc']}: {err['msg']}" for err in errors]

    logger.warning(
        f"Validation error: {error_messages}",
        extra={"path": request.url.path, "method": request.method, "errors": errors},
    )

    validation_error = ValidationError(
        message="Request validation failed",
        code=ErrorCodes.A_VALIDATION_001,
        details={"validation_errors": errors},
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"success": False, "error": validation_error.to_dict()},
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""

    logger.exception(
        f"Unexpected error: {exc}",
        exc_info=True,
        extra={"path": request.url.path, "method": request.method},
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "error_code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred",
                "details": {},
            },
        },
    )
