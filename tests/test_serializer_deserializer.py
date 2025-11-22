"""
serializer_deserializer 模块的单元测试
测试基于 Pydantic 的序列化/反序列化功能
"""
import pytest
from typing import Dict, Any, List
from llama_index.core.schema import Document

# 添加项目根目录到路径
import sys
import os
project_root = os.path.join(os.path.dirname(__file__), '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from service.parser.serializer_deserializer import (
    DocumentModel,
    to_dict,
    to_dict_list,
    from_dict,
    from_dict_list,
    to_json,
    to_json_list,
    from_json,
    from_json_list,
    serialize_document,
    serialize_documents,
    deserialize_document,
    deserialize_documents
)


class TestDocumentModel:
    """测试 DocumentModel Pydantic 模型"""
    
    def test_from_document(self):
        """测试从 Document 创建 DocumentModel"""
        doc = Document(text="测试文档", metadata={"author": "Ellen", "type": "test"})
        model = DocumentModel.from_document(doc)
        
        assert model.text == "测试文档"
        assert model.metadata["author"] == "Ellen"
        assert model.metadata["type"] == "test"
    
    def test_from_document_with_id(self):
        """测试带 ID 的 Document"""
        doc = Document(text="test", metadata={})
        doc.id_ = "doc-123"
        model = DocumentModel.from_document(doc)
        
        assert model.id == "doc-123"
    
    def test_from_document_without_id(self):
        """测试没有 ID 的 Document"""
        doc = Document(text="test", metadata={})
        # llama_index 的 Document 会自动生成 ID，所以我们需要验证 ID 存在
        model = DocumentModel.from_document(doc)
        
        # Document 会自动生成 UUID，所以 ID 不应该是 None
        assert model.id is not None
        assert isinstance(model.id, str)
    
    def test_to_document(self):
        """测试转换为 Document"""
        model = DocumentModel(text="测试", metadata={"key": "value"}, id="123")
        doc = model.to_document()
        
        assert doc.text == "测试"
        assert doc.metadata["key"] == "value"
        assert doc.id_ == "123"
    
    def test_to_dict(self):
        """测试转换为字典"""
        model = DocumentModel(text="test", metadata={"key": "value"})
        result = model.to_dict()
        
        assert isinstance(result, dict)
        assert result["text"] == "test"
        assert result["metadata"]["key"] == "value"
    
    def test_to_json(self):
        """测试转换为 JSON"""
        model = DocumentModel(text="测试", metadata={"author": "Ellen"})
        json_str = model.to_json()
        
        assert isinstance(json_str, str)
        assert "测试" in json_str
        assert "Ellen" in json_str
    
    def test_text_validation_none(self):
        """测试 text 为 None 时自动转换为空字符串"""
        model = DocumentModel(text=None, metadata={})
        assert model.text == ""
    
    def test_text_validation_number(self):
        """测试 text 为数字时自动转换为字符串"""
        model = DocumentModel(text=123, metadata={})
        assert model.text == "123"
        assert isinstance(model.text, str)
    
    def test_metadata_validation_none(self):
        """测试 metadata 为 None 时自动转换为空字典"""
        model = DocumentModel(text="test", metadata=None)
        assert model.metadata == {}
    
    def test_metadata_validation_invalid_type(self):
        """测试 metadata 类型无效时抛出异常"""
        with pytest.raises(ValueError):
            DocumentModel(text="test", metadata="invalid")


class TestSerializationFunctions:
    """测试序列化函数"""
    
    def test_to_dict_basic(self):
        """测试基本序列化"""
        doc = Document(text="测试文档", metadata={"author": "Ellen"})
        result = to_dict(doc)
        
        assert result["text"] == "测试文档"
        assert result["metadata"]["author"] == "Ellen"
    
    def test_to_dict_with_empty_text(self):
        """测试空文本"""
        doc = Document(text="", metadata={})
        result = to_dict(doc)
        
        assert result["text"] == ""
        assert result["metadata"] == {}
    
    def test_to_dict_with_none_values(self):
        """测试 None 值"""
        # llama_index 的 Document 不接受 metadata=None，所以测试空值情况
        doc = Document(text="", metadata={})
        result = to_dict(doc)
        
        assert result["text"] == ""
        assert result["metadata"] == {}
    
    def test_to_dict_invalid_type(self):
        """测试无效类型抛出异常"""
        with pytest.raises(TypeError):
            to_dict("not a document")
    
    def test_to_dict_list_basic(self):
        """测试列表序列化"""
        docs = [
            Document(text="文档1", metadata={"id": 1}),
            Document(text="文档2", metadata={"id": 2}),
            Document(text="文档3", metadata={"id": 3})
        ]
        result = to_dict_list(docs)
        
        assert len(result) == 3
        assert result[0]["text"] == "文档1"
        assert result[1]["metadata"]["id"] == 2
        assert result[2]["text"] == "文档3"
    
    def test_to_dict_list_empty(self):
        """测试空列表"""
        result = to_dict_list([])
        assert result == []
    
    def test_to_json_basic(self):
        """测试 JSON 序列化"""
        doc = Document(text="测试", metadata={"key": "value"})
        json_str = to_json(doc)
        
        assert isinstance(json_str, str)
        assert "测试" in json_str
        assert "value" in json_str
    
    def test_to_json_with_indent(self):
        """测试带缩进的 JSON"""
        doc = Document(text="test", metadata={})
        json_str = to_json(doc, indent=2)
        
        assert "\n" in json_str  # 缩进会包含换行
    
    def test_to_json_with_ensure_ascii(self):
        """测试 ASCII 编码"""
        doc = Document(text="测试", metadata={})
        json_str = to_json(doc, ensure_ascii=True)
        
        assert "\\u" in json_str  # Unicode 转义
    
    def test_to_json_list_basic(self):
        """测试 JSON 列表序列化"""
        docs = [
            Document(text="doc1", metadata={}),
            Document(text="doc2", metadata={})
        ]
        json_str = to_json_list(docs)
        
        assert isinstance(json_str, str)
        assert "doc1" in json_str
        assert "doc2" in json_str


class TestDeserializationFunctions:
    """测试反序列化函数"""
    
    def test_from_dict_basic(self):
        """测试基本反序列化"""
        data = {
            "text": "测试文档",
            "metadata": {"author": "Ellen"},
            "id": "doc-123"
        }
        doc = from_dict(data)
        
        assert doc.text == "测试文档"
        assert doc.metadata["author"] == "Ellen"
        assert doc.id_ == "doc-123"
    
    def test_from_dict_with_id_field(self):
        """测试支持 id_ 字段名"""
        data = {
            "text": "test",
            "metadata": {},
            "id_": "doc-456"
        }
        doc = from_dict(data)
        
        assert doc.id_ == "doc-456"
    
    def test_from_dict_without_id(self):
        """测试没有 ID 的情况"""
        data = {
            "text": "test",
            "metadata": {}
        }
        doc = from_dict(data)
        
        assert doc.text == "test"
        assert doc.metadata == {}
    
    def test_from_dict_invalid_type(self):
        """测试无效类型"""
        with pytest.raises(TypeError):
            from_dict("not a dict")
    
    def test_from_dict_invalid_data(self):
        """测试无效数据 - Pydantic 会忽略额外字段，测试缺少必需字段"""
        with pytest.raises(ValueError):
            # metadata 字段类型错误会导致验证失败
            from_dict({"text": "test", "metadata": "not a dict"})
    
    def test_from_dict_list_basic(self):
        """测试列表反序列化"""
        data_list = [
            {"text": "doc1", "metadata": {"id": 1}},
            {"text": "doc2", "metadata": {"id": 2}},
            {"text": "doc3", "metadata": {"id": 3}}
        ]
        docs = from_dict_list(data_list)
        
        assert len(docs) == 3
        assert docs[0].text == "doc1"
        assert docs[1].metadata["id"] == 2
        assert docs[2].text == "doc3"
    
    def test_from_dict_list_empty(self):
        """测试空列表"""
        result = from_dict_list([])
        assert result == []
    
    def test_from_dict_list_invalid_type(self):
        """测试无效类型"""
        with pytest.raises(TypeError):
            from_dict_list("not a list")
    
    def test_from_json_basic(self):
        """测试 JSON 反序列化"""
        json_str = '{"text": "测试", "metadata": {"key": "value"}, "id": null}'
        doc = from_json(json_str)
        
        assert doc.text == "测试"
        assert doc.metadata["key"] == "value"
    
    def test_from_json_invalid(self):
        """测试无效 JSON"""
        with pytest.raises(ValueError):
            from_json("invalid json")
    
    def test_from_json_list_basic(self):
        """测试 JSON 列表反序列化"""
        json_str = '[{"text": "doc1", "metadata": {}}, {"text": "doc2", "metadata": {}}]'
        docs = from_json_list(json_str)
        
        assert len(docs) == 2
        assert docs[0].text == "doc1"
        assert docs[1].text == "doc2"
    
    def test_from_json_list_invalid(self):
        """测试无效 JSON 列表"""
        with pytest.raises(ValueError):
            from_json_list("invalid json")


class TestRoundTripSerialization:
    """测试序列化和反序列化的往返转换"""
    
    def test_round_trip_single_document(self):
        """测试单个文档的往返转换"""
        original = Document(text="测试文档", metadata={"author": "Ellen", "year": 2025})
        
        # 序列化
        serialized = to_dict(original)
        
        # 反序列化
        restored = from_dict(serialized)
        
        assert restored.text == original.text
        assert restored.metadata == original.metadata
    
    def test_round_trip_document_list(self):
        """测试文档列表的往返转换"""
        originals = [
            Document(text=f"文档{i}", metadata={"id": i})
            for i in range(5)
        ]
        
        # 序列化
        serialized = to_dict_list(originals)
        
        # 反序列化
        restored = from_dict_list(serialized)
        
        assert len(restored) == len(originals)
        for i, doc in enumerate(restored):
            assert doc.text == originals[i].text
            assert doc.metadata == originals[i].metadata
    
    def test_round_trip_with_json(self):
        """测试 JSON 往返转换"""
        original = Document(text="测试", metadata={"中文": "支持"})
        
        # 序列化为 JSON
        json_str = to_json(original)
        
        # 从 JSON 反序列化
        restored = from_json(json_str)
        
        assert restored.text == original.text
        assert restored.metadata == original.metadata


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    def test_serialize_document(self):
        """测试 serialize_document"""
        doc = Document(text="test", metadata={"key": "value"})
        result = serialize_document(doc)
        
        assert result["text"] == "test"
        assert result["metadata"]["key"] == "value"
    
    def test_serialize_documents(self):
        """测试 serialize_documents"""
        docs = [
            Document(text="doc1", metadata={}),
            Document(text="doc2", metadata={})
        ]
        result = serialize_documents(docs)
        
        assert len(result) == 2
        assert result[0]["text"] == "doc1"
    
    def test_deserialize_document(self):
        """测试 deserialize_document"""
        data = {"text": "test", "metadata": {}}
        doc = deserialize_document(data)
        
        assert doc.text == "test"
    
    def test_deserialize_documents(self):
        """测试 deserialize_documents"""
        data_list = [
            {"text": "doc1", "metadata": {}},
            {"text": "doc2", "metadata": {}}
        ]
        docs = deserialize_documents(data_list)
        
        assert len(docs) == 2
        assert docs[0].text == "doc1"


class TestEdgeCases:
    """测试边界情况"""
    
    def test_empty_document(self):
        """测试空文档"""
        doc = Document(text="", metadata={})
        serialized = serialize_document(doc)
        deserialized = deserialize_document(serialized)
        
        assert deserialized.text == ""
        assert deserialized.metadata == {}
    
    def test_large_metadata(self):
        """测试大型元数据"""
        metadata = {f"key_{i}": f"value_{i}" for i in range(1000)}
        doc = Document(text="test", metadata=metadata)
        
        serialized = serialize_document(doc)
        deserialized = deserialize_document(serialized)
        
        assert len(deserialized.metadata) == 1000
        assert deserialized.metadata["key_500"] == "value_500"
    
    def test_nested_metadata(self):
        """测试嵌套元数据"""
        metadata = {
            "level1": {
                "level2": {
                    "level3": "deep value"
                }
            }
        }
        doc = Document(text="test", metadata=metadata)
        
        serialized = serialize_document(doc)
        deserialized = deserialize_document(serialized)
        
        assert deserialized.metadata["level1"]["level2"]["level3"] == "deep value"
    
    def test_special_characters_in_text(self):
        """测试文本中的特殊字符"""
        special_text = "Line1\nLine2\tTabbed\r\nWindows\\Path\"Quoted\""
        doc = Document(text=special_text, metadata={})
        
        serialized = serialize_document(doc)
        deserialized = deserialize_document(serialized)
        
        assert deserialized.text == special_text
    
    def test_unicode_in_metadata(self):
        """测试元数据中的 Unicode"""
        metadata = {
            "中文": "测试",
            "français": "test",
            "emoji": "🎉✅"
        }
        doc = Document(text="test", metadata=metadata)
        
        serialized = serialize_document(doc)
        deserialized = deserialize_document(serialized)
        
        assert deserialized.metadata["中文"] == "测试"
        assert deserialized.metadata["français"] == "test"
        assert deserialized.metadata["emoji"] == "🎉✅"
    
    def test_very_long_text(self):
        """测试超长文本"""
        long_text = "a" * 100000  # 100K 字符
        doc = Document(text=long_text, metadata={})
        
        serialized = serialize_document(doc)
        deserialized = deserialize_document(serialized)
        
        assert len(deserialized.text) == 100000
        assert deserialized.text == long_text
