"""
ContentExtractor 模块的单元测试
"""
import pytest
from unittest.mock import Mock, MagicMock, patch, call
from typing import List, Optional
from llama_index.core.schema import Document

# 添加项目根目录到路径
import sys
import os
project_root = os.path.join(os.path.dirname(__file__), '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from service.parser.extractor import ContentExtractor
from service.parser.base import BaseParser
from service.parser.registry import ParserRegistry


class MockParser(BaseParser):
    """用于测试的 Mock 解析器"""
    
    supported_extensions = ['.mock']
    supported_mime_types = ['application/mock']
    
    def can_parse(self, filename: str, content_type: Optional[str]) -> bool:
        return filename.endswith('.mock') or (content_type and 'mock' in content_type.lower())
    
    def extract_content(self, contents: bytes, filename: str, content_type: Optional[str]) -> List[Document]:
        return [Document(text="mock content", metadata={"file_name": filename})]


class TestContentExtractor:
    """测试 ContentExtractor 类"""
    
    def test_init(self):
        """测试初始化"""
        extractor = ContentExtractor()
        assert extractor.registry is not None
        assert isinstance(extractor.registry, ParserRegistry)
    
    def test_auto_register(self):
        """测试自动注册解析器"""
        extractor = ContentExtractor()
        
        # 检查解析器是否已注册
        parsers = extractor.registry.list_parsers()
        assert len(parsers) >= 2  # 至少应该有 TextParser 和 PDFParser
        assert 'TextParser' in parsers
        assert 'PDFParser' in parsers
    
    def test_extract_success(self):
        """测试成功提取文件内容"""
        extractor = ContentExtractor()
        
        # 注册 Mock 解析器用于测试
        extractor.register_parser(MockParser)
        
        # 测试提取
        contents = b"mock file content"
        documents = extractor.extract(contents, "test.mock", "application/mock")
        
        assert documents is not None
        assert len(documents) == 1
        assert documents[0].text == "mock content"
        assert documents[0].metadata["file_name"] == "test.mock"
    
    def test_extract_unsupported_file_type(self):
        """测试不支持的文件类型"""
        extractor = ContentExtractor()
        
        # 测试不支持的文件类型
        with pytest.raises(ValueError) as exc_info:
            extractor.extract(b"content", "test.unknown", "application/unknown")
        
        assert "unsupported file type" in str(exc_info.value).lower()
        assert "test.unknown" in str(exc_info.value)
    
    def test_extract_parser_returns_empty_list(self):
        """测试解析器返回空列表的情况"""
        extractor = ContentExtractor()
        extractor.register_parser(MockParser)
        
        # Mock 解析器返回空列表
        mock_parser = Mock(spec=BaseParser)
        mock_parser.extract_content.return_value = []
        extractor.registry._extension_map['.mock'] = lambda: mock_parser
        
        documents = extractor.extract(b"content", "test.mock")
        assert documents == []
    
    def test_extract_parser_raises_exception(self):
        """测试解析器抛出异常的情况"""
        extractor = ContentExtractor()
        extractor.register_parser(MockParser)
        
        # Mock 解析器抛出异常
        mock_parser = Mock(spec=BaseParser)
        mock_parser.extract_content.side_effect = ValueError("parsing error")
        extractor.registry._extension_map['.mock'] = lambda: mock_parser
        
        with pytest.raises(ValueError) as exc_info:
            extractor.extract(b"content", "test.mock")
        
        assert "parsing error" in str(exc_info.value)
    
    def test_extract_with_content_type(self):
        """测试使用 content_type 提取"""
        extractor = ContentExtractor()
        extractor.register_parser(MockParser)
        
        documents = extractor.extract(
            b"content",
            "test.unknown",
            content_type="application/mock"
        )
        
        assert documents is not None
        assert len(documents) == 1
    
    def test_register_parser(self):
        """测试动态注册解析器"""
        extractor = ContentExtractor()
        initial_count = len(extractor.registry.list_parsers())
        
        # 注册新的解析器
        extractor.register_parser(MockParser)
        
        # 检查解析器已注册
        parsers = extractor.registry.list_parsers()
        assert len(parsers) == initial_count + 1
        assert 'MockParser' in parsers
        
        # 检查可以使用新注册的解析器
        documents = extractor.extract(b"content", "test.mock")
        assert documents is not None
    
    def test_register_parser_multiple_times(self):
        """测试多次注册同一个解析器"""
        extractor = ContentExtractor()
        extractor.register_parser(MockParser)
        initial_count = len(extractor.registry.list_parsers())
        
        # 再次注册同一个解析器
        extractor.register_parser(MockParser)
        
        # 应该不会增加数量（会覆盖）
        parsers = extractor.registry.list_parsers()
        assert len(parsers) == initial_count
    
    def test_extract_logging(self):
        """测试日志记录"""
        extractor = ContentExtractor()
        extractor.register_parser(MockParser)
        
        with patch('service.parser.extractor.logger') as mock_logger:
            extractor.extract(b"content", "test.mock")
            
            # 检查日志调用
            assert mock_logger.info.called
            # 检查是否记录了开始提取的日志
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            assert any('extract' in str(call).lower() for call in mock_logger.info.call_args_list)
    
    def test_extract_error_logging(self):
        """测试错误日志记录"""
        extractor = ContentExtractor()
        extractor.register_parser(MockParser)
        
        # Mock 解析器抛出异常
        mock_parser = Mock(spec=BaseParser)
        mock_parser.extract_content.side_effect = ValueError("test error")
        extractor.registry._extension_map['.mock'] = lambda: mock_parser
        
        with patch('service.parser.extractor.logger') as mock_logger:
            with pytest.raises(ValueError):
                extractor.extract(b"content", "test.mock")
            
            # 检查错误日志
            assert mock_logger.error.called
            error_calls = [str(call) for call in mock_logger.error.call_args_list]
            assert any('failed' in str(call).lower() for call in mock_logger.error.call_args_list)
    
    def test_extract_with_empty_content(self):
        """测试提取空内容"""
        extractor = ContentExtractor()
        extractor.register_parser(MockParser)
        
        documents = extractor.extract(b"", "test.mock")
        assert documents is not None
        assert len(documents) == 1
    
    def test_extract_with_none_content_type(self):
        """测试 content_type 为 None 的情况"""
        extractor = ContentExtractor()
        extractor.register_parser(MockParser)
        
        # 只通过扩展名匹配
        documents = extractor.extract(b"content", "test.mock", content_type=None)
        assert documents is not None
    
    def test_extract_with_empty_filename(self):
        """测试空文件名"""
        extractor = ContentExtractor()
        
        with pytest.raises(ValueError):
            extractor.extract(b"content", "", None)


class TestContentExtractorIntegration:
    """集成测试：使用真实的解析器"""
    
    def test_extract_text_file(self):
        """测试提取文本文件"""
        extractor = ContentExtractor()
        
        # 测试 .txt 文件
        contents = b"Hello, World!"
        documents = extractor.extract(contents, "test.txt", "text/plain")
        
        assert documents is not None
        assert len(documents) == 1
        assert documents[0].text == "Hello, World!"
        assert documents[0].metadata["file_name"] == "test.txt"
    
    def test_extract_markdown_file(self):
        """测试提取 Markdown 文件"""
        extractor = ContentExtractor()
        
        contents = b"# Title\n\nSome content"
        documents = extractor.extract(contents, "test.md", "text/markdown")
        
        assert documents is not None
        assert len(documents) == 1
        assert "# Title" in documents[0].text
    
    def test_extract_with_mime_type_only(self):
        """测试仅通过 MIME 类型匹配"""
        extractor = ContentExtractor()
        
        # 使用不常见的扩展名但正确的 MIME 类型
        contents = b"Text content"
        documents = extractor.extract(contents, "test.unknown", "text/plain")
        
        assert documents is not None
        assert len(documents) == 1


class TestContentExtractorEdgeCases:
    """边界情况测试"""
    
    def test_extract_with_very_long_filename(self):
        """测试超长文件名"""
        extractor = ContentExtractor()
        extractor.register_parser(MockParser)
        
        long_filename = "a" * 1000 + ".mock"
        documents = extractor.extract(b"content", long_filename)
        assert documents is not None
    
    def test_extract_with_special_characters_in_filename(self):
        """测试文件名包含特殊字符"""
        extractor = ContentExtractor()
        extractor.register_parser(MockParser)
        
        special_filename = "test-file_123 (copy).mock"
        documents = extractor.extract(b"content", special_filename)
        assert documents is not None
    
    def test_extract_with_multiple_dots_in_filename(self):
        """测试文件名包含多个点"""
        extractor = ContentExtractor()
        extractor.register_parser(MockParser)
        
        multi_dot_filename = "test.file.name.mock"
        documents = extractor.extract(b"content", multi_dot_filename)
        assert documents is not None
    
    def test_extract_with_path_in_filename(self):
        """测试文件名包含路径"""
        extractor = ContentExtractor()
        extractor.register_parser(MockParser)
        
        path_filename = "/path/to/test.mock"
        documents = extractor.extract(b"content", path_filename)
        assert documents is not None
    
    def test_extract_returns_multiple_documents(self):
        """测试解析器返回多个文档"""
        extractor = ContentExtractor()
        
        # 创建返回多个文档的 Mock 解析器
        mock_parser = Mock(spec=BaseParser)
        mock_parser.extract_content.return_value = [
            Document(text="doc1", metadata={"page": 1}),
            Document(text="doc2", metadata={"page": 2}),
            Document(text="doc3", metadata={"page": 3})
        ]
        extractor.registry._extension_map['.mock'] = lambda: mock_parser
        
        documents = extractor.extract(b"content", "test.mock")
        assert len(documents) == 3
        assert documents[0].text == "doc1"
        assert documents[1].text == "doc2"
        assert documents[2].text == "doc3"

