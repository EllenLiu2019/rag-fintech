from typing import Any, Optional
from pydantic import BaseModel


class APIResponse(BaseModel):
    """API response model"""

    success: bool
    data: Optional[Any] = None
    error: Optional[dict] = None

    @classmethod
    def ok(cls, data: Any = None):
        return cls(success=True, data=data)

    @classmethod
    def fail(cls, error_code: str, message: str, details: dict = None):
        return cls(
            success=False,
            error={
                "error_code": error_code,
                "message": message,
                "details": details or {},
            },
        )
