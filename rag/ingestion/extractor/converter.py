import re
import logging
from typing import Any, Optional, Callable

logger = logging.getLogger(__name__)


class FieldConvert:
    """
    Field value converter

    Convert the extracted raw field values to standard format, e.g.:
    - Number fields: remove commas, currency symbols, convert to numeric type
    - Date fields: standardize date format
    - String fields: clean and standardize
    """

    def __init__(self) -> None:
        self.convert_functions: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "number": self.convert_number,
            "string": self.convert_string,
        }

    def convert(self, extracted_result: dict[str, Any]) -> None:
        """
        该方法会递归处理提取结果，对每个字段应用相应的转换函数。
        转换结果会直接修改到原始字典中（添加 convert_value 字段）。

        Example:
        extracted_result:
            {
                "field_name": {
                    "type": "number",
                    "raw_value": "2,000,000",
                    "transform": {"type": ["remove_commas"]},
                    "convert_value": "2000000"
                    ...
                },
                "list_field": [
                    {"sub_field": {...}},
                    ...
                ]
            }

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
        Apply the corresponding conversion function to the field value

        Args:
            property_value: field value dictionary, must contain "type" key
        """
        field_type = property_value.get("type")
        if not field_type:
            return

        convert_func = self.convert_functions.get(field_type, self.convert_string)

        try:
            convert_func(property_value)
        except Exception as e:
            logger.warning(f"Failed to convert field of type '{field_type}': {e}", exc_info=True)

    def convert_number(self, property_value: dict[str, Any]) -> dict[str, Any]:
        """
        Parse and convert number field

        Supported conversion operations:
        - remove_commas: remove commas
        - remove_currency: remove currency symbols and spaces

        Args:
            property_value: field value dictionary, must contain:
                - raw_value: original string value
                - transform: conversion configuration, format: {"type": ["remove_commas", ...]}

        Returns:
            dict[str, Any]: modified field value dictionary, added convert_value field
        """
        if not isinstance(property_value, dict):
            logger.warning(f"Expected dict for number conversion, got {type(property_value).__name__}")
            return property_value

        raw_value = property_value.get("raw_value", "")
        if not raw_value:
            logger.debug("No raw_value found for number conversion")
            return property_value

        transform = property_value.get("transform", {})
        transform_types: list[str] = transform.get("type", [])

        if not transform_types:
            logger.debug("No transform types specified, skipping conversion")
            return property_value

        convert_value = self._apply_number_transforms(raw_value, transform_types)

        numeric_value = self._try_convert_to_numeric(convert_value)

        property_value["convert_value"] = numeric_value if numeric_value is not None else convert_value

        logger.debug(f"Converted number: '{raw_value}' -> '{property_value['convert_value']}'")

        return property_value

    def _apply_number_transforms(self, value: str, transform_types: list[str]) -> str:
        """
        Apply number conversion operations

        Args:
            value: original string value
            transform_types: conversion type list

        Returns:
            str: converted string
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
                    logger.warning(f"Failed to apply transform '{transform_type}': {e}", exc_info=True)
            else:
                logger.warning(f"Unknown transform type: {transform_type}")

        return converted_value

    def _try_convert_to_numeric(self, value: str) -> Optional[float]:
        if not value:
            return None

        try:
            numeric_value = float(value)
            if numeric_value.is_integer():
                return float(int(numeric_value))
            return numeric_value
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not convert '{value}' to numeric: {e}")
            return None

    def convert_string(self, property_value: dict[str, Any]) -> dict[str, Any]:

        if not isinstance(property_value, dict):
            return property_value

        raw_value = property_value.get("raw_value", "")

        if isinstance(raw_value, str):
            cleaned_value = raw_value.strip()
            if cleaned_value != raw_value:
                property_value["convert_value"] = cleaned_value

        return property_value

    def register_converter(self, field_type: str, converter_func: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        if not callable(converter_func):
            raise ValueError(f"converter_func must be callable, got {type(converter_func).__name__}")

        self.convert_functions[field_type] = converter_func
        logger.info(f"Registered custom converter for type: {field_type}")

