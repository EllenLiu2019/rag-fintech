"""
应用配置管理模块
使用 Pydantic Settings 管理配置，支持环境变量
"""

from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration class"""

    # ========== API ==========
    API_TITLE: str = Field(default="RAG FinTech API", description="API title")
    API_DESCRIPTION: str = Field(
        default="RAG FinTech backend API", description="API description"
    )
    API_VERSION: str = Field(default="1.0.0", description="API version")

    # ========== CORS ==========
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:5173"], description="Allowed CORS origins")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, description="Allow CORS credentials")
    CORS_ALLOW_METHODS: List[str] = Field(default=["*"], description="Allowed HTTP methods")
    CORS_ALLOW_HEADERS: List[str] = Field(default=["*"], description="Allowed HTTP headers")

    # ========== Logging ==========
    LOG_LEVEL: str = Field(default="INFO", description="Log level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s", description="Log format"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from environment variable (comma-separated string) or list."""
        if isinstance(self.CORS_ORIGINS, str):
            return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
        return self.CORS_ORIGINS


settings = Settings()
