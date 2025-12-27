from typing import Any, Optional, Union, TypedDict

from common import get_logger

logger = get_logger(__name__)


class FieldConfig(TypedDict, total=False):
    """Field configuration definition"""

    type: str
    mapping: Union[str, tuple[str, ...]]  # Field mapping path (source_key, chinese_field_name)


class MetadataCreator:

    def __init__(self) -> None:
        self.schema: dict[str, FieldConfig] = self._load_schema()

    @classmethod
    def _load_schema(cls) -> dict[str, FieldConfig]:
        """
        Schema mapping definition.

        mapping format:
        - str: direct key access, get first value from the dict
          e.g. "policy_number" → extracted_result["policy_number"] → {"保单号": "xxx"} → "xxx"
        - tuple: (source_key, chinese_field_name)
          e.g. ("policy_holder", "投保人") → extracted_result["policy_holder"]["投保人"]
        """
        return {
            "policy_number": {"type": "str", "mapping": "policy_number"},
            "holder_name": {"type": "str", "mapping": ("policy_holder", "投保人")},
            "holder_gender": {"type": "str", "mapping": ("policy_holder", "性别")},
            "holder_birth_date": {"type": "str", "mapping": ("policy_holder", "出生日期")},
            "holder_id_number": {"type": "str", "mapping": ("policy_holder", "证件号码")},
            "insured_name": {"type": "str", "mapping": ("insured", "被保险人")},
            "insured_gender": {"type": "str", "mapping": ("insured", "性别")},
            "insured_birth_date": {"type": "str", "mapping": ("insured", "出生日期")},
            "insured_id_number": {"type": "str", "mapping": ("insured", "证件号码")},
            "insured_relationship_to_holder": {"type": "str", "mapping": ("insured", "与投保人关系")},
            "effective_date": {"type": "str", "mapping": "effective_date"},
            "expiry_date": {"type": "str", "mapping": "expiry_date"},
        }

    def create(self, extracted_result: dict[str, Any]) -> dict[str, Any]:
        """
        根据 schema 中定义的字段，从 extracted_result 中提取对应的值。
        优先使用 convert_value（如果存在），否则使用 raw_value。

        Args:
            extracted_result:
                {
                    "policy_number": {"保单号": "AO1234567890FE"},
                    "policy_holder": {"投保人": "张三", "性别": "男", "出生日期": "1990-01-01", "证件号码": "123456789012345678"},
                    "insured": {"被保险人": "李四", "性别": "女", "出生日期": "1990-01-01", "证件号码": "123456789012345678"},
                    "coverage": [{"保险名称": "个人癌症医疗保险（互联网2022版A款）", "保险责任": "恶性肿瘤质子重离子医疗保险金", "最高保险金额（元）": "2,000,000", "详细说明": "首次投保或非连续投保等待期:90天<br/>免赔额:0元/年<br/>社保目录内医疗费用赔付比例:100%<br/>社保目录外医疗费用赔付比例:100%"}, ...],
                    "cvg_premium": [{"条款名称": "个人癌症医疗保险（互联网2022版A款）", "保险费（元）": "2,284.00"}, ...],
                    "effective_date": {"保险期间开始日期": "2025-01-01"},
                    "expiry_date": {"保险期间结束日期": "2026-01-01"},
                }

        Returns:
            dict[str, Any]: 元数据字典，格式为：
                {
                    "policy_number": "AO1234567890FE",
                    "holder_name": "张三",
                    "holder_gender": "男",
                    "hodler_birth_date": "1990-01-01",
                    "hodler_id_number": "123456789012345678",
                    "insured_name": "李四",
                    "insured_gender": "女",
                    "insured_birth_date": "1990-01-01",
                    "insured_id_number": "123456789012345678",
                    "insured_relationship_to_holder": "父母",
                    "effective_date": "2025-01-01",
                    "expiry_date": "2026-01-01",
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
            logger.debug(f"Metadata: {metadata}")
            return metadata

        except Exception as e:
            logger.error(f"Error creating metadata: {e}", exc_info=True)
            return metadata

    def _extract_field_value(self, extracted_result: dict[str, Any], field_name: str) -> Optional[Any]:
        """
        从 extracted_result 中提取字段值

        支持：
        - 直接字段：policy_number → extracted_result["policy_number"] → 取第一个值
        - 嵌套字段：holder_name → extracted_result["policy_holder"]["投保人"]

        Args:
            extracted_result: 提取结果字典
            field_name: 元数据字段名

        Returns:
            Optional[Any]: 提取的字段值，如果不存在则返回 None
        """
        field_config = self.schema.get(field_name)
        if not field_config:
            logger.warning(f"No config found for field: {field_name}")
            return None

        mapping = field_config.get("mapping")
        if not mapping:
            logger.warning(f"No mapping found for field: {field_name}")
            return None

        try:
            # 直接映射 (str): 从 extracted_result[key] 取第一个值
            if isinstance(mapping, str):
                source_data = extracted_result.get(mapping)
                if source_data is None:
                    return None
                # 如果是列表，直接返回
                if isinstance(source_data, list):
                    return source_data
                # 如果是字典，取第一个值
                if isinstance(source_data, dict):
                    return next(iter(source_data.values()), None)
                return source_data

            # 嵌套映射 (tuple): (source_key, chinese_field_name)
            elif isinstance(mapping, tuple) and len(mapping) == 2:
                source_key, field_key = mapping
                source_data = extracted_result.get(source_key)
                if not isinstance(source_data, dict):
                    return None
                return source_data.get(field_key)

            else:
                logger.warning(f"Invalid mapping type for field '{field_name}': {type(mapping)}")
                return None

        except (KeyError, TypeError, IndexError) as e:
            logger.debug(f"Error extracting field '{field_name}': {e}")
            return None

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
