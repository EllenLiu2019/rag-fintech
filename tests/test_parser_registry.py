"""
ParserRegistry 模块的单元测试
"""
import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Optional
from llama_index.core.schema import Document

# 添加项目根目录到路径
import sys
import os
project_root = os.path.join(os.path.dirname(__file__), '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from service.parser.registry import ParserRegistry
from service.parser.base import BaseParser


class MockParser(BaseParser):
    """用于测试的 Mock 解析器"""
    
    supported_extensions = ['.mock']
    supported_mime_types = ['application/mock']
    
    def can_parse(self, filename: str, content_type: Optional[str]) -> bool:
        return filename.endswith('.mock') or (content_type and 'mock' in content_type.lower())
    
    def extract_content(self, contents: bytes, filename: str, content_type: Optional[str]) -> List[Document]:
        return [Document(text="mock content", metadata={"file_name": filename})]


class MockParserMultipleExts(BaseParser):
    """支持多个扩展名的 Mock 解析器"""
    
    supported_extensions = ['.ext1', '.ext2', '.EXT3']  # 测试大小写
    supported_mime_types = ['application/ext1', 'application/ext2']
    
    def can_parse(self, filename: str, content_type: Optional[str]) -> bool:
        return any(filename.lower().endswith(ext.lower()) for ext in self.supported_extensions)
    
    def extract_content(self, contents: bytes, filename: str, content_type: Optional[str]) -> List[Document]:
        return [Document(text="mock content", metadata={"file_name": filename})]


class TestParserRegistry:
    """测试 ParserRegistry 类"""
    
    def test_init(self):
        """测试初始化"""
        registry = ParserRegistry()
        assert registry._parsers == {}
        assert registry._extension_map == {}
        assert registry._mime_map == {}
    
    def test_register_parser(self):
        """测试注册解析器"""
        registry = ParserRegistry()
        registry.register(MockParser)
        
        # 检查解析器已注册
        assert MockParser.__name__ in registry._parsers
        assert registry._parsers[MockParser.__name__] == MockParser
        
        # 检查扩展名映射
        assert '.mock' in registry._extension_map
        assert registry._extension_map['.mock'] == MockParser
        
        # 检查 MIME 类型映射
        assert 'application/mock' in registry._mime_map
        assert registry._mime_map['application/mock'] == MockParser
    
    def test_register_parser_multiple_extensions(self):
        """测试注册支持多个扩展名的解析器"""
        registry = ParserRegistry()
        registry.register(MockParserMultipleExts)
        
        # 检查所有扩展名都已注册（应该转换为小写）
        assert '.ext1' in registry._extension_map
        assert '.ext2' in registry._extension_map
        assert '.ext3' in registry._extension_map  # 大写转换为小写
        assert registry._extension_map['.ext1'] == MockParserMultipleExts
        assert registry._extension_map['.ext2'] == MockParserMultipleExts
        assert registry._extension_map['.ext3'] == MockParserMultipleExts
        
        # 检查所有 MIME 类型都已注册
        assert 'application/ext1' in registry._mime_map
        assert 'application/ext2' in registry._mime_map
    
    def test_register_multiple_parsers(self):
        """测试注册多个解析器"""
        registry = ParserRegistry()
        registry.register(MockParser)
        registry.register(MockParserMultipleExts)
        
        # 检查两个解析器都已注册
        assert len(registry._parsers) == 2
        assert MockParser.__name__ in registry._parsers
        assert MockParserMultipleExts.__name__ in registry._parsers
    
    def test_get_parser_by_extension(self):
        """测试通过扩展名获取解析器"""
        registry = ParserRegistry()
        registry.register(MockParser)
        
        # 测试匹配扩展名
        parser = registry.get_parser("test.mock")
        assert parser is not None
        assert isinstance(parser, MockParser)
        
        # 测试大小写不敏感
        parser = registry.get_parser("test.MOCK")
        assert parser is not None
        assert isinstance(parser, MockParser)
    
    def test_get_parser_by_extension_multiple_exts(self):
        """测试通过多个扩展名获取解析器"""
        registry = ParserRegistry()
        registry.register(MockParserMultipleExts)
        
        # 测试不同的扩展名
        parser1 = registry.get_parser("test.ext1")
        assert parser1 is not None
        assert isinstance(parser1, MockParserMultipleExts)
        
        parser2 = registry.get_parser("test.ext2")
        assert parser2 is not None
        assert isinstance(parser2, MockParserMultipleExts)
        
        parser3 = registry.get_parser("test.ext3")
        assert parser3 is not None
        assert isinstance(parser3, MockParserMultipleExts)
    
    def test_get_parser_by_mime_type(self):
        """测试通过 MIME 类型获取解析器"""
        registry = ParserRegistry()
        registry.register(MockParser)
        
        # 测试匹配 MIME 类型
        parser = registry.get_parser("test.unknown", content_type="application/mock")
        assert parser is not None
        assert isinstance(parser, MockParser)
        
        # 测试大小写不敏感
        parser = registry.get_parser("test.unknown", content_type="APPLICATION/MOCK")
        assert parser is not None
        assert isinstance(parser, MockParser)
    
    def test_get_parser_mime_type_with_parameters(self):
        """测试 MIME 类型带参数的情况（如 charset）"""
        registry = ParserRegistry()
        registry.register(MockParser)
        
        # 测试带参数的 MIME 类型
        parser = registry.get_parser("test.unknown", content_type="application/mock; charset=utf-8")
        assert parser is not None
        assert isinstance(parser, MockParser)
    
    def test_get_parser_extension_priority(self):
        """测试扩展名优先于 MIME 类型"""
        registry = ParserRegistry()
        registry.register(MockParser)
        
        # 扩展名匹配应该优先
        parser = registry.get_parser("test.mock", content_type="application/other")
        assert parser is not None
        assert isinstance(parser, MockParser)
    
    def test_get_parser_not_found(self):
        """测试找不到解析器的情况"""
        registry = ParserRegistry()
        registry.register(MockParser)
        
        # 测试未注册的扩展名和 MIME 类型
        parser = registry.get_parser("test.unknown", content_type="application/unknown")
        assert parser is None
    
    def test_get_parser_no_extension(self):
        """测试没有扩展名的文件"""
        registry = ParserRegistry()
        registry.register(MockParser)
        
        # 测试没有扩展名的文件
        parser = registry.get_parser("testfile")
        assert parser is None or isinstance(parser, MockParser)  # 如果 MIME 类型匹配
    
    def test_get_parser_empty_filename(self):
        """测试空文件名"""
        registry = ParserRegistry()
        registry.register(MockParser)
        
        parser = registry.get_parser("")
        assert parser is None
    
    def test_list_parsers(self):
        """测试列出所有解析器"""
        registry = ParserRegistry()
        
        # 初始状态应该为空
        assert registry.list_parsers() == []
        
        # 注册解析器后
        registry.register(MockParser)
        parsers = registry.list_parsers()
        assert len(parsers) == 1
        assert MockParser.__name__ in parsers
        
        # 注册多个解析器
        registry.register(MockParserMultipleExts)
        parsers = registry.list_parsers()
        assert len(parsers) == 2
        assert MockParser.__name__ in parsers
        assert MockParserMultipleExts.__name__ in parsers
    
    def test_register_same_parser_twice(self):
        """测试重复注册同一个解析器（应该覆盖）"""
        registry = ParserRegistry()
        registry.register(MockParser)
        registry.register(MockParser)  # 再次注册
        
        # 应该只有一个解析器
        assert len(registry._parsers) == 1
        assert MockParser.__name__ in registry._parsers
    
    def test_get_parser_returns_new_instance(self):
        """测试每次获取解析器都返回新实例"""
        registry = ParserRegistry()
        registry.register(MockParser)
        
        parser1 = registry.get_parser("test.mock")
        parser2 = registry.get_parser("test.mock")
        
        # 应该返回不同的实例
        assert parser1 is not None
        assert parser2 is not None
        assert parser1 is not parser2
        assert isinstance(parser1, MockParser)
        assert isinstance(parser2, MockParser)
    
    def test_get_parser_with_complex_filename(self):
        """测试复杂文件名（多个点）"""
        registry = ParserRegistry()
        registry.register(MockParser)
        
        # 测试多个点的文件名
        parser = registry.get_parser("test.file.mock")
        assert parser is not None
        assert isinstance(parser, MockParser)
        
        # 测试路径中的文件名
        parser = registry.get_parser("/path/to/test.mock")
        assert parser is not None
        assert isinstance(parser, MockParser)
    
    def test_get_parser_mime_type_case_insensitive(self):
        """测试 MIME 类型大小写不敏感"""
        registry = ParserRegistry()
        registry.register(MockParser)
        
        # 测试各种大小写组合
        test_cases = [
            "APPLICATION/MOCK",
            "application/MOCK",
            "APPLICATION/mock",
            "Application/Mock"
        ]
        
        for mime_type in test_cases:
            parser = registry.get_parser("test.unknown", content_type=mime_type)
            assert parser is not None, f"Failed for {mime_type}"
            assert isinstance(parser, MockParser)


class TestParserRegistryIntegration:
    """集成测试：使用真实的解析器"""
    
    def test_register_text_parser(self):
        """测试注册真实的 TextParser"""
        from service.parser.text_parser import TextParser
        
        registry = ParserRegistry()
        registry.register(TextParser)
        
        # 检查扩展名映射
        assert '.txt' in registry._extension_map
        assert '.md' in registry._extension_map
        assert '.markdown' in registry._extension_map
        
        # 检查 MIME 类型映射
        assert 'text/plain' in registry._mime_map
        assert 'text/markdown' in registry._mime_map
        
        # 测试获取解析器
        parser = registry.get_parser("test.txt")
        assert parser is not None
        assert isinstance(parser, TextParser)
        
        parser = registry.get_parser("test.md")
        assert parser is not None
        assert isinstance(parser, TextParser)
        
        parser = registry.get_parser("test.unknown", content_type="text/plain")
        assert parser is not None
        assert isinstance(parser, TextParser)

