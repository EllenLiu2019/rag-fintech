from abc import ABC, abstractmethod
from typing import Optional, List

from llama_index.core.schema import Document

class BaseParser(ABC):
    """文件解析器抽象基类"""
    
    # 支持的文件扩展名列表
    supported_extensions: List[str] = []
    
    # 支持的 MIME 类型列表
    supported_mime_types: List[str] = []
    
    @abstractmethod
    def can_parse(self, filename: str, content_type: Optional[str]) -> bool:
        """判断是否可以解析该文件"""
        pass
    
    @abstractmethod
    def parse_content(self, contents: bytes, filename: str, content_type: Optional[str]) -> List[Document]:
        """解析文本内容"""
        pass
    
    def get_metadata(self, contents: bytes, filename: str) -> dict:
        """获取文件元数据（可选）"""
        return {}