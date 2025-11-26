"""
应用配置管理模块
使用 Pydantic Settings 管理配置，支持环境变量
"""
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application configuration class"""
    
    # ========== API 基本信息 ==========
    API_TITLE: str = Field(default="RAG FinTech API", description="API 标题")
    API_DESCRIPTION: str = Field(
        default="RAG 智能洞察后端 API - 提供文件上传、解析和内容提取服务",
        description="API 描述"
    )
    API_VERSION: str = Field(default="1.0.0", description="API 版本")
    
    # ========== 服务器配置 ==========
    HOST: str = Field(default="0.0.0.0", description="服务器主机地址")
    PORT: int = Field(default=8001, description="服务器端口")
    DEBUG: bool = Field(default=False, description="调试模式")
    
    # ========== CORS configuration ==========
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:5173"],
        description="允许的 CORS 源地址列表"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True, description="允许 CORS 凭证")
    CORS_ALLOW_METHODS: List[str] = Field(
        default=["*"],
        description="允许的 HTTP 方法"
    )
    CORS_ALLOW_HEADERS: List[str] = Field(
        default=["*"],
        description="允许的 HTTP 头"
    )
    
    # ========== File upload configuration ==========
    MAX_FILE_SIZE_MB: int = Field(default=3, description="最大文件大小（MB）")
    MAX_FILE_SIZE_BYTES: int = Field(default=3 * 1024 * 1024, description="最大文件大小（字节）")
    UPLOAD_DIR: str = Field(default="uploaded_files", description="上传文件存储目录")
    ALLOWED_FILE_EXTENSIONS: List[str] = Field(
        default=[".txt", ".pdf", ".doc", ".docx", ".md"],
        description="允许的文件扩展名"
    )
    
    # ========== Logging configuration ==========
    LOG_LEVEL: str = Field(default="INFO", description="日志级别")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s",
        description="日志格式"
    )
    
    # ========== Database configuration (reserved) ==========
    # DATABASE_URL: str = Field(default="sqlite:///./app.db", description="Database connection URL")
    
    class Config:
        """Pydantic configuration"""
        env_file = ".env"  # 从 .env 文件读取配置
        env_file_encoding = "utf-8"
        case_sensitive = False  # 环境变量不区分大小写
        
    @property
    def max_file_size_bytes(self) -> int:
        """计算最大文件大小（字节）"""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    def get_cors_origins(self) -> List[str]:
        """获取 CORS 源地址列表（支持从环境变量解析）"""
        if isinstance(self.CORS_ORIGINS, str):
            # 如果是从环境变量读取的字符串，按逗号分割
            return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
        return self.CORS_ORIGINS


# 创建全局配置实例
settings = Settings()

