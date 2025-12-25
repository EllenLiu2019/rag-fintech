from datetime import datetime
from decimal import Decimal
from typing import Any, Optional, Callable

from common import get_logger

logger = get_logger(__name__)


class FieldConvert:
    """
    Field value converter
    """

    def __init__(self) -> None:
        self.converters: dict[str, Callable[[Any], Any]] = {
            "string": self.convert_string,
            "date": self.convert_date,
            "decimal": self.convert_decimal,
            "float": self.convert_float,
        }

    def convert(self, value: Any, value_type: str) -> Any:
        if not value:
            logger.warning("Value is None, cannot convert")
            return
        try:
            convert_func = self.converters.get(value_type, self.convert_string)
            return convert_func(value)
        except Exception as e:
            logger.error(f"Error converting value: {e}", exc_info=True)
            raise

    def convert_date(self, date_str: str) -> Optional[datetime]:
        if not date_str:
            return None

        if not isinstance(date_str, str):
            return date_str

        date_str = date_str.replace("年", "-").replace("月", "-").replace("日", "").strip()

        formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Failed to parse date: {date_str}")
        return None

    def convert_decimal(self, value: Any) -> Optional[Decimal]:
        if value is None:
            return None

        if isinstance(value, Decimal):
            return value

        if isinstance(value, (int, float)):
            try:
                return Decimal(str(value))
            except (ValueError, Exception) as e:
                logger.warning(f"Failed to convert number to decimal: {value}, error: {e}")
                return None

        if isinstance(value, str):
            if not value.strip():
                return None
            try:
                cleaned = value.replace(",", "").replace(" ", "").strip()
                return Decimal(cleaned)
            except (ValueError, Exception) as e:
                logger.warning(f"Failed to parse decimal string: {value}, error: {e}")
                return None

        try:
            return Decimal(str(value))
        except (ValueError, Exception) as e:
            logger.warning(f"Failed to parse decimal: {value}, error: {e}")
            return None

    def convert_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None

        if isinstance(value, float):
            return value

        if isinstance(value, int):
            return float(value)

        if isinstance(value, str):
            if not value.strip():
                return None
            try:
                cleaned = value.replace(",", "").replace(" ", "").strip()
                return float(cleaned)
            except (ValueError, Exception) as e:
                logger.warning(f"Failed to parse float string: {value}, error: {e}")
                return None

        try:
            return float(value)
        except (ValueError, Exception) as e:
            logger.warning(f"Failed to parse float: {value}, error: {e}")
            return None

    def convert_string(self, value: Any) -> Any:
        if not isinstance(value, str):
            return value
        return value.strip()

    def _get_value(self, entity_data: dict, key: str, default: Any = None) -> Any:
        if key in entity_data:
            value, _ = entity_data[key]
            return value
        return default


field_converter = FieldConvert()
