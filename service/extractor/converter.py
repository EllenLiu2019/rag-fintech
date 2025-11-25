import re
import logging
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)


class FieldConvert:
    """
    字段值转换器
    
    负责将提取的原始字段值转换为标准格式，例如：
    - 数字字段：移除逗号、货币符号，转换为数值类型
    - 日期字段：标准化日期格式
    - 字符串字段：清理和标准化
    """
    
    def __init__(self) -> None:
        """
        初始化字段转换器
        
        注册所有可用的转换函数
        """
        self.convert_functions: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "number": self.convert_number,
            "string": self.convert_string,
            # 可以扩展更多类型：date, datetime, boolean 等
        }

    def convert(self, extracted_result: dict[str, Any]) -> None:
        """
        转换提取结果中的所有字段值
        
        该方法会递归处理提取结果，对每个字段应用相应的转换函数。
        转换结果会直接修改到原始字典中（添加 convert_value 字段）。
        
        Args:
            extracted_result: 提取结果字典，格式为：
                {
                    "field_name": {
                        "type": "number",
                        "raw_value": "2,000,000",
                        "transform": {"type": ["remove_commas"]},
                        ...
                    },
                    "list_field": [
                        {"sub_field": {...}},
                        ...
                    ]
                }
                
        Note:
            该方法会直接修改输入的字典，不返回新字典
        """
        if not isinstance(extracted_result, dict):
            logger.warning(f"Expected dict, got {type(extracted_result).__name__}")
            return
        
        try:
            for _, property_value in extracted_result.items():
                if isinstance(property_value, list):
                    self._convert_list_field(property_value)
                elif isinstance(property_value, dict):
                    self._convert_dict_field(property_value)
        except Exception as e:
            logger.error(f"Error converting extracted result: {e}", exc_info=True)
            raise

    def _convert_list_field(self, property_list: list[dict[str, Any]]) -> None:
        """
        转换列表类型的字段
        
        Args:
            property_list: 包含多个字典的列表
        """
        for prop in property_list:
            if not isinstance(prop, dict):
                continue
            for _, sub_value in prop.items():
                if not isinstance(sub_value, dict) or not sub_value.get("type"):
                    continue
                self._apply_conversion(sub_value)

    def _convert_dict_field(self, property_value: dict[str, Any]) -> None:
        """
        转换字典类型的字段
        
        Args:
            property_value: 字段值字典
        """
        if not isinstance(property_value, dict) or not property_value.get("type"):
            return
        self._apply_conversion(property_value)

    def _apply_conversion(self, property_value: dict[str, Any]) -> None:
        """
        应用相应的转换函数到字段值
        
        Args:
            property_value: 字段值字典，必须包含 "type" 键
        """
        field_type = property_value.get("type")
        if not field_type:
            return
        
        convert_func = self.convert_functions.get(
            field_type, 
            self.convert_string  # 默认使用字符串转换
        )
        
        try:
            convert_func(property_value)
        except Exception as e:
            logger.warning(
                f"Failed to convert field of type '{field_type}': {e}",
                exc_info=True
            )

    def convert_number(self, property_value: dict[str, Any]) -> dict[str, Any]:
        """
        解析和转换数字字段
        
        支持的转换操作：
        - remove_commas: 移除千位分隔符
        - remove_currency: 移除货币符号和空格
        
        Args:
            property_value: 字段值字典，包含：
                - raw_value: 原始字符串值
                - transform: 转换配置，格式为 {"type": ["remove_commas", ...]}
                
        Returns:
            dict[str, Any]: 修改后的字段值字典，添加了 convert_value 字段
        """
        if not isinstance(property_value, dict):
            logger.warning(f"Expected dict for number conversion, got {type(property_value).__name__}")
            return property_value
        
        raw_value = property_value.get("raw_value", "")
        if not raw_value:
            logger.debug("No raw_value found for number conversion")
            return property_value
        
        # 获取转换配置
        transform = property_value.get("transform", {})
        transform_types: list[str] = transform.get("type", [])
        
        if not transform_types:
            logger.debug("No transform types specified, skipping conversion")
            return property_value
        
        # 应用转换操作
        convert_value = self._apply_number_transforms(raw_value, transform_types)
        
        # 尝试转换为数值类型
        numeric_value = self._try_convert_to_numeric(convert_value)
        
        # 更新字段值
        property_value["convert_value"] = numeric_value if numeric_value is not None else convert_value
        
        logger.debug(
            f"Converted number: '{raw_value}' -> '{property_value['convert_value']}'"
        )
        
        return property_value

    def _apply_number_transforms(self, value: str, transform_types: list[str]) -> str:
        """
        应用数字转换操作
        
        Args:
            value: 原始字符串值
            transform_types: 转换类型列表
            
        Returns:
            str: 转换后的字符串
        """
        convert_methods: dict[str, Callable[[str], str]] = {
            "remove_commas": lambda x: x.replace(",", ""),
            "remove_currency": lambda x: re.sub(r"[¥$€£元,\s]", "", str(x)),
        }
        
        converted_value = str(value)
        
        for transform_type in transform_types:
            if transform_type in convert_methods:
                try:
                    converted_value = convert_methods[transform_type](converted_value)
                except Exception as e:
                    logger.warning(
                        f"Failed to apply transform '{transform_type}': {e}",
                        exc_info=True
                    )
            else:
                logger.warning(f"Unknown transform type: {transform_type}")
        
        return converted_value

    def _try_convert_to_numeric(self, value: str) -> Optional[float]:
        """
        尝试将字符串转换为数值
        
        Args:
            value: 要转换的字符串
            
        Returns:
            Optional[float]: 转换后的数值，如果转换失败则返回 None
        """
        if not value:
            return None
        
        try:
            # 尝试转换为浮点数
            numeric_value = float(value)
            # 如果是整数，返回整数形式（但类型仍然是 float）
            if numeric_value.is_integer():
                return float(int(numeric_value))
            return numeric_value
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not convert '{value}' to numeric: {e}")
            return None
    
    def convert_string(self, property_value: dict[str, Any]) -> dict[str, Any]:
        """
        转换字符串字段（默认处理）
        
        字符串字段通常不需要特殊转换，但可以在这里添加清理逻辑：
        - 去除首尾空格
        - 标准化空白字符
        - 移除特殊字符等
        
        Args:
            property_value: 字段值字典
            
        Returns:
            dict[str, Any]: 原字段值字典（可能已添加 convert_value）
        """
        if not isinstance(property_value, dict):
            return property_value
        
        raw_value = property_value.get("raw_value", "")
        
        # 基本清理：去除首尾空格
        if isinstance(raw_value, str):
            cleaned_value = raw_value.strip()
            if cleaned_value != raw_value:
                property_value["convert_value"] = cleaned_value
        
        return property_value

    def register_converter(
        self, 
        field_type: str, 
        converter_func: Callable[[dict[str, Any]], dict[str, Any]]
    ) -> None:
        """
        注册自定义转换函数
        
        Args:
            field_type: 字段类型名称
            converter_func: 转换函数，接受字段值字典，返回修改后的字典
        """
        if not callable(converter_func):
            raise ValueError(f"converter_func must be callable, got {type(converter_func).__name__}")
        
        self.convert_functions[field_type] = converter_func
        logger.info(f"Registered custom converter for type: {field_type}")
