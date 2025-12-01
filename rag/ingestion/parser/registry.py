from typing import Dict, Type, Optional, List
from .base import BaseParser

class ParserRegistry:
    """解析器注册表"""
    
    def __init__(self):
        self._parsers: Dict[str, Type[BaseParser]] = {}
        self._extension_map: Dict[str, Type[BaseParser]] = {}
        self._mime_map: Dict[str, Type[BaseParser]] = {}
    
    def register(self, parser_class: Type[BaseParser]):
        """注册解析器"""
        parser = parser_class()
        
        # 按扩展名注册
        for ext in parser.supported_extensions:
            self._extension_map[ext.lower()] = parser_class
        
        # 按 MIME 类型注册
        for mime in parser.supported_mime_types:
            self._mime_map[mime.lower()] = parser_class
        
        self._parsers[parser_class.__name__] = parser_class
    
    def get_parser(self, filename: str, content_type: Optional[str] = None) -> Optional[BaseParser]:
        """根据文件名和类型获取解析器"""
        # 优先按扩展名匹配
        if '.' in filename:
            ext = '.' + filename.split('.')[-1].lower()  # 添加点号以匹配注册时的格式
            if ext in self._extension_map:
                return self._extension_map[ext]()
        
        # 按 MIME 类型匹配
        if content_type:
            mime = content_type.lower().split(';')[0].strip()
            if mime in self._mime_map:
                return self._mime_map[mime]()
        
        return None
    
    def list_parsers(self) -> List[str]:
        """列出所有已注册的解析器"""
        return list(self._parsers.keys())

