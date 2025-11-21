from .base import BaseParser
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class TextParser(BaseParser):
    """文本文件解析器"""
    
    supported_extensions = ['.txt', '.md', '.markdown']
    supported_mime_types = ['text/plain', 'text/markdown', 'text/x-markdown']
    
    def can_parse(self, filename: str, content_type: Optional[str]) -> bool:
        """判断是否可以解析该文件"""
        # 按扩展名判断
        ext_match = filename.lower().endswith(tuple(self.supported_extensions))
        
        # 按 MIME 类型判断
        mime_match = False
        if content_type:
            content_type_lower = content_type.lower()
            mime_match = any(mime in content_type_lower for mime in self.supported_mime_types) or \
                        'text' in content_type_lower
        
        return ext_match or mime_match
    
    def extract_content(self, contents: bytes, filename: str, content_type: Optional[str]) -> str:
        """提取文本内容"""
        try:
            return contents.decode('utf-8', errors='ignore')
        except Exception as e:
            error_msg = f"text file decoding failed: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

