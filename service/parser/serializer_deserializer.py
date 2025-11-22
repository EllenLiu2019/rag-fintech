"""
文档序列化工具模块
使用 Pydantic 实现类型安全的序列化/反序列化
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator
from llama_index.core.schema import Document
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic 模型定义
# ============================================================================

class DocumentModel(BaseModel):
    """
    Document 的 Pydantic 模型
    用于类型安全的序列化和反序列化
    """
    model_config = ConfigDict(
        # 允许任意类型（兼容 llama_index 的 Document）
        arbitrary_types_allowed=True,
        # JSON 序列化时排除 None 值
        exclude_none=False,
        # 验证赋值
        validate_assignment=True
    )
    
    text: str = Field(default="", description="文档文本内容")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="文档元数据")
    id: Optional[str] = Field(default=None, description="文档唯一标识符")
    
    @field_validator('text', mode='before')
    @classmethod
    def validate_text(cls, v):
        """确保 text 始终是字符串"""
        if v is None:
            return ""
        return str(v)
    
    @field_validator('metadata', mode='before')
    @classmethod
    def validate_metadata(cls, v):
        """确保 metadata 始终是字典"""
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError(f"metadata must be a dict, got {type(v).__name__}")
        return v
    
    @classmethod
    def from_document(cls, doc: Document) -> "DocumentModel":
        """
        从 llama_index Document 对象创建 Pydantic 模型
        
        Args:
            doc: Document 对象
            
        Returns:
            DocumentModel: Pydantic 模型实例
        """
        if not isinstance(doc, Document):
            raise TypeError(f"Expected Document, got {type(doc).__name__}")
        
        # 安全地提取 id
        doc_id = None
        if hasattr(doc, 'id_'):
            try:
                doc_id = str(doc.id_) if doc.id_ else None
            except (AttributeError, TypeError):
                doc_id = None
        
        return cls(
            text=doc.text or "",
            metadata=doc.metadata or {},
            id=doc_id
        )
    
    def to_document(self) -> Document:
        """
        转换为 llama_index Document 对象
        
        Returns:
            Document: Document 对象
        """
        doc = Document(
            text=self.text,
            metadata=self.metadata
        )
        
        # 如果有 id，设置它
        if self.id:
            doc.id_ = self.id
        
        return doc
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典（使用 Pydantic 的 model_dump）
        
        Returns:
            dict: 序列化后的字典
        """
        return self.model_dump(exclude_none=False)
    
    def to_json(self, indent: Optional[int] = None, ensure_ascii: bool = False) -> str:
        """
        转换为 JSON 字符串（使用 Pydantic 的 model_dump_json）
        
        Args:
            indent: JSON 缩进
            ensure_ascii: 是否确保 ASCII 编码
            
        Returns:
            str: JSON 字符串
        """
        return self.model_dump_json(
            indent=indent,
            exclude_none=False
        )


# ============================================================================
# 模块级函数：Document <-> Dict/JSON
# ============================================================================

def to_dict(doc: Document) -> Dict[str, Any]:
    """
    将 Document 对象转换为字典
    
    Args:
        doc: Document 对象
        
    Returns:
        dict: 包含 text、metadata 和 id 的字典
        
    Raises:
        TypeError: 如果 doc 不是 Document 对象
    """
    return DocumentModel.from_document(doc).to_dict()


def to_dict_list(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    将 Document 对象列表转换为字典列表
    
    Args:
        documents: Document 对象列表
        
    Returns:
        list: 字典列表
    """
    if not documents:
        return []
    
    return [to_dict(doc) for doc in documents]


def to_json(doc: Document, indent: Optional[int] = None, ensure_ascii: bool = False) -> str:
    """
    将 Document 对象序列化为 JSON 字符串
    
    Args:
        doc: Document 对象
        indent: JSON 缩进，None 表示紧凑格式
        ensure_ascii: 是否确保 ASCII 编码（默认 False，支持中文）
        
    Returns:
        str: JSON 字符串
    """
    model = DocumentModel.from_document(doc)
    if ensure_ascii:
        # 使用标准 json 库以支持 ensure_ascii
        return json.dumps(model.to_dict(), ensure_ascii=ensure_ascii, indent=indent)
    return model.to_json(indent=indent)


def to_json_list(documents: List[Document], indent: Optional[int] = None, ensure_ascii: bool = False) -> str:
    """
    将 Document 对象列表序列化为 JSON 字符串
    
    Args:
        documents: Document 对象列表
        indent: JSON 缩进
        ensure_ascii: 是否确保 ASCII 编码（默认 False，支持中文）
        
    Returns:
        str: JSON 字符串
    """
    dict_list = to_dict_list(documents)
    return json.dumps(dict_list, ensure_ascii=ensure_ascii, indent=indent)


def from_dict(data: Dict[str, Any]) -> Document:
    """
    从字典反序列化为 Document 对象
    
    Args:
        data: 包含 text、metadata 和可选的 id 的字典
        
    Returns:
        Document: Document 对象
        
    Raises:
        TypeError: 如果 data 不是字典
        ValueError: 如果数据格式不正确
    """
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict, got {type(data).__name__}")
    
    try:
        # 处理 id 字段（支持 "id" 和 "id_" 两种键名）
        if "id_" in data and "id" not in data:
            data["id"] = data["id_"]
        
        # 使用 Pydantic 验证和解析
        model = DocumentModel(**data)
        return model.to_document()
    except Exception as e:
        logger.error(f"Failed to deserialize document from dict: {data}, error: {e}")
        raise ValueError(f"Invalid document data: {e}") from e


def from_dict_list(data_list: List[Dict[str, Any]]) -> List[Document]:
    """
    从字典列表反序列化为 Document 对象列表
    
    Args:
        data_list: 字典列表
        
    Returns:
        list: Document 对象列表
        
    Raises:
        TypeError: 如果 data_list 不是列表
    """
    if not isinstance(data_list, list):
        raise TypeError(f"Expected list, got {type(data_list).__name__}")
    
    if not data_list:
        return []
    
    return [from_dict(data) for data in data_list]


def from_json(json_str: str) -> Document:
    """
    从 JSON 字符串反序列化为 Document 对象
    
    Args:
        json_str: JSON 字符串
        
    Returns:
        Document: Document 对象
        
    Raises:
        ValueError: 如果 JSON 格式不正确
    """
    try:
        data = json.loads(json_str)
        return from_dict(data)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}, error: {e}")
        raise ValueError(f"Invalid JSON string: {e}") from e


def from_json_list(json_str: str) -> List[Document]:
    """
    从 JSON 字符串反序列化为 Document 对象列表
    
    Args:
        json_str: JSON 字符串
        
    Returns:
        list: Document 对象列表
        
    Raises:
        ValueError: 如果 JSON 格式不正确
    """
    try:
        data_list = json.loads(json_str)
        return from_dict_list(data_list)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON: {json_str}, error: {e}")
        raise ValueError(f"Invalid JSON string: {e}") from e


# ============================================================================
# 便捷函数（保持向后兼容）
# ============================================================================

def serialize_document(doc: Document) -> Dict[str, Any]:
    """
    序列化单个 Document 对象
    
    Args:
        doc: Document 对象
        
    Returns:
        dict: 序列化后的字典
    """
    return to_dict(doc)


def serialize_documents(documents: List[Document]) -> List[Dict[str, Any]]:
    """
    序列化 Document 对象列表
    
    Args:
        documents: Document 对象列表
        
    Returns:
        list: 序列化后的字典列表
    """
    return to_dict_list(documents)


def deserialize_document(data: Dict[str, Any]) -> Document:
    """
    反序列化单个 Document 对象
    
    Args:
        data: 包含文档数据的字典
        
    Returns:
        Document: Document 对象
    """
    return from_dict(data)


def deserialize_documents(data_list: List[Dict[str, Any]]) -> List[Document]:
    """
    反序列化 Document 对象列表
    
    Args:
        data_list: 字典列表
        
    Returns:
        list: Document 对象列表
    """
    return from_dict_list(data_list)


