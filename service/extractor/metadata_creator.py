import logging
from typing import Any, Optional, Union, TypedDict

logger = logging.getLogger(__name__)


class FieldConfig(TypedDict, total=False):
    """Field configuration definition"""

    type: str
    mapping: Union[str, tuple[str, ...]]  # Field mapping path


class MetadataCreator:

    def __init__(self) -> None:
        self.schema: dict[str, FieldConfig] = self._load_schema()

    @classmethod
    def _load_schema(cls) -> dict[str, FieldConfig]:
        return {
            "document_id": {"type": "str", "mapping": "document_id"},
            "policy_number": {"type": "str", "mapping": "policy_number"},
            "holder_name": {"type": "str", "mapping": ("policy_holder", "name")},
            "hodler_gender": {"type": "str", "mapping": ("policy_holder", "gender")},
            "hodler_birth_date": {"type": "str", "mapping": ("policy_holder", "birth_date")},
            "hodler_id_number": {"type": "str", "mapping": ("policy_holder", "id_number")},
            "insured_name": {"type": "str", "mapping": ("insured", "name")},
            "insured_gender": {"type": "str", "mapping": ("insured", "gender")},
            "insured_birth_date": {"type": "str", "mapping": ("insured", "birth_date")},
            "insured_id_number": {"type": "str", "mapping": ("insured", "id_number")},
            "insured_relationship_to_holder": {"type": "str", "mapping": ("insured", "relationship_to_holder")},
            "effective_date": {"type": "str", "mapping": "effective_date"},
            "expiry_date": {"type": "str", "mapping": "expiry_date"},
            "coverage": {"type": "list", "mapping": "coverage"},  # Preserve complex structure
            "cvg_premium": {"type": "list", "mapping": "cvg_premium"},
        }

    def create(self, extracted_result: dict[str, Any]) -> dict[str, Any]:
        """
        根据 schema 中定义的字段，从 extracted_result 中提取对应的值。
        优先使用 convert_value（如果存在），否则使用 raw_value。

        Args:
            extracted_result:
                {
                    "policy_number": {
                        "type": "string",
                        "raw_value": "ABC123",
                        "convert_value": "ABC123"  # 可选
                    },
                    "policy_holder": {
                        "name": {
                            "type": "string",
                            "raw_value": "张三"
                        },
                        ...
                    },
                    ...
                }

        Returns:
            dict[str, Any]: 元数据字典，格式为：
                {
                    "document_id": "...",
                    "policy_number": "ABC123",
                    "holder_name": "张三",
                    "insured_name": "李四"
                }
        """
        if not isinstance(extracted_result, dict):
            logger.warning(f"Expected dict for extracted_result, got {type(extracted_result).__name__}")
            return {}

        metadata: dict[str, Any] = {}

        try:
            # 遍历 schema 中定义的所有字段
            for field_name, field_config in self.schema.items():
                value = self._extract_field_value(extracted_result, field_name)
                if value is not None:
                    metadata[field_name] = value
                else:
                    logger.debug(f"Field '{field_name}' not found in extracted_result")

            logger.info(f"Created metadata with {len(metadata)} fields")

            embed_parts = self._collect_embed_parts(extracted_result)
            if embed_parts:
                metadata["embed_metadata"] = "; ".join(embed_parts)

            logger.info(f"Metadata: {metadata}")
            return metadata

        except Exception as e:
            logger.error(f"Error creating metadata: {e}", exc_info=True)
            return metadata

    def _extract_field_value(self, extracted_result: dict[str, Any], field_name: str) -> Optional[Any]:
        """
        从 extracted_result 中提取字段值

        支持：
        - 直接字段：policy_number → extracted_result["policy_number"]
        - 嵌套字段：holder_name → extracted_result["policy_holder"]["name"]
        - 列表字段：coverage → extracted_result["coverage"]
        - 列表字段：cvg_premium → extracted_result["cvg_premium"]
        Args:
            extracted_result: 提取结果字典
            field_name: 元数据字段名

        Returns:
            Optional[Any]: 提取的字段值，如果不存在则返回 None
        """
        # 获取字段配置
        field_config = self.schema.get(field_name)
        if not field_config:
            logger.warning(f"No config found for field: {field_name}")
            return None

        # 获取字段映射路径
        mapping = field_config.get("mapping")
        if not mapping:
            logger.warning(f"No mapping found for field: {field_name}")
            return None

        try:
            # 处理直接映射
            if isinstance(mapping, str):
                return self._extract_value_from_path(extracted_result, [mapping])

            # 处理嵌套映射（元组）
            elif isinstance(mapping, tuple):
                return self._extract_value_from_path(extracted_result, list(mapping))

            else:
                logger.warning(f"Invalid mapping type for field '{field_name}': {type(mapping)}")
                return None

        except (KeyError, TypeError, IndexError) as e:
            logger.debug(f"Error extracting field '{field_name}': {e}")
            return None

    def _collect_embed_parts(self, data: Any) -> list[str]:
        """
        Recursively collect embed_text and value pairs from extracted result.
        """
        parts = []
        if isinstance(data, list):
            for item in data:
                parts.extend(self._collect_embed_parts(item))
        elif isinstance(data, dict):
            # Check if it's a leaf node (contains raw_value)
            if "raw_value" in data:
                embed_text = data.get("embed_text")
                if embed_text:
                    val = data.get("convert_value")
                    if val is None:
                        val = data.get("raw_value")

                    if val is not None:
                        parts.append(f"{embed_text}:{val}")
            else:
                # It's a container, recurse on values
                for value in data.values():
                    parts.extend(self._collect_embed_parts(value))
        return parts

    def _extract_value_from_path(self, data: dict[str, Any], path: list[str]) -> Optional[Any]:
        """
        从嵌套字典中按路径提取值，并递归清洗数据（剥离 type, raw_value 等元信息）
        """
        if not path:
            return None

        # 按路径导航到目标字段
        current = data
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]

        return self._clean_value(current)

    def _clean_value(self, value: Any) -> Any:
        """
        递归清洗值：
        1. 如果是包含 raw_value/convert_value 的提取对象，解包取值。
        2. 如果是列表，递归清洗每一项。
        3. 如果是字典，递归清洗每一个 value。
        """
        if isinstance(value, list):
            return [self._clean_value(item) for item in value]

        if isinstance(value, dict):
            # Case 1: 它是一个提取对象 (包含 raw_value 或 convert_value)
            if "raw_value" in value or "convert_value" in value:
                if "convert_value" in value and value["convert_value"] is not None:
                    return value["convert_value"]
                return value.get("raw_value")

            # Case 2: 普通字典，递归清洗其内容
            return {k: self._clean_value(v) for k, v in value.items()}

        return value

    def register_field_mapping(self, field_name: str, mapping: Union[str, tuple[str, ...]]) -> None:
        """
        注册或更新字段映射

        Args:
            field_name: 元数据字段名
            mapping: 字段映射路径，可以是字符串（直接映射）或元组（嵌套映射）
        """
        if not isinstance(mapping, (str, tuple)):
            raise ValueError(f"mapping must be str or tuple, got {type(mapping).__name__}")

        # 如果字段已存在，更新映射；否则创建新字段（使用默认类型 "str"）
        if field_name in self.schema:
            self.schema[field_name]["mapping"] = mapping
        else:
            self.schema[field_name] = {"type": "str", "mapping": mapping}  # 默认类型

        logger.info(f"Registered field mapping: {field_name} -> {mapping}")

    def add_schema_field(
        self, field_name: str, field_type: str, mapping: Optional[Union[str, tuple[str, ...]]] = None
    ) -> None:
        """
        添加新的 schema 字段

        Args:
            field_name: 元数据字段名
            field_type: 字段类型
            mapping: 可选的字段映射，如果不提供则使用 field_name 作为映射
        """
        if mapping is None:
            mapping = field_name

        self.schema[field_name] = {"type": field_type, "mapping": mapping}

        logger.info(f"Added schema field: {field_name} ({field_type}) with mapping: {mapping}")

    def get_field_type(self, field_name: str) -> Optional[str]:
        """
        获取字段类型

        Args:
            field_name: 元数据字段名

        Returns:
            Optional[str]: 字段类型，如果字段不存在则返回 None
        """
        field_config = self.schema.get(field_name)
        return field_config.get("type") if field_config else None

    def get_field_mapping(self, field_name: str) -> Optional[Union[str, tuple[str, ...]]]:
        """
        获取字段映射路径

        Args:
            field_name: 元数据字段名

        Returns:
            Optional[Union[str, tuple[str, ...]]]: 字段映射路径，如果字段不存在则返回 None
        """
        field_config = self.schema.get(field_name)
        return field_config.get("mapping") if field_config else None
