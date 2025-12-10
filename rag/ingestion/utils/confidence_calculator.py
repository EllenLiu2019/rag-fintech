import re
import logging
from typing import Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class FieldConfidence:
    """Confidence scores for a single field"""

    field_key: str
    confidence: float = 0.0
    source_confidence: float = 0.0
    completeness: float = 0.0
    value_quality: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class ConfidenceReport:
    """Overall confidence report"""

    overall_confidence: float = 0.0
    field_confidences: dict[str, FieldConfidence] = field(default_factory=dict)
    schema_completeness: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_confidence": self.overall_confidence,
            "field_confidence": {
                k: {
                    "confidence": v.confidence,
                    "source_confidence": v.source_confidence,
                    "completeness": v.completeness,
                    "value_quality": v.value_quality,
                    "warnings": v.warnings,
                }
                for k, v in self.field_confidences.items()
            },
            "schema_completeness": self.schema_completeness,
            "warnings": self.warnings,
            "errors": self.errors,
        }


class ConfidenceCalculator:
    """
    Confidence calculator for the new extraction structure.

    The new structure is:
    {
        "policy_number": {"保单号": "AO1234567890FE"},
        "policy_holder": {"投保人": "张三", "性别": "男", ...},
        "coverage": [{"保险名称": "...", ...}, ...],
    }

    Confidence dimensions:
    1. Source confidence: How reliable is the extraction source (header match vs grid fallback)
    2. Completeness: How many fields were extracted for this entity
    3. Value quality: Are the values non-empty and well-formatted
    """

    # Expected minimum fields for object-type entities
    EXPECTED_FIELDS = {
        "policy_holder": ["投保人", "性别", "出生日期", "证件号码"],
        "insured": ["被保险人", "性别", "出生日期", "证件号码"],
    }

    # Field type validators
    FIELD_VALIDATORS = {
        "出生日期": "date",
        "性别": "gender",
        "证件号码": "id_number",
        "手机号码": "phone",
        "电子邮箱": "email",
        "保险期间开始日期": "date",
        "保险期间结束日期": "date",
    }

    def calculate(
        self,
        extracted_result: dict[str, Any],
        schema: dict[str, Any],
        extraction_signals: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """
        Calculate confidence scores for extraction results.

        Args:
            extracted_result: The extracted data
            schema: The schema definition
            extraction_signals: Optional signals from extraction process

        Returns:
            Confidence report as dictionary
        """
        if extraction_signals is None:
            extraction_signals = {}

        report = ConfidenceReport()
        expected_fields = set(schema.get("fields", {}).keys())

        # Calculate confidence for each extracted field
        for field_key, field_data in extracted_result.items():
            field_conf = self._calculate_field_confidence(field_key, field_data, extraction_signals.get(field_key, {}))
            report.field_confidences[field_key] = field_conf

            # Collect warnings
            if field_conf.warnings:
                report.warnings.extend([f"{field_key}: {w}" for w in field_conf.warnings])

            if field_conf.confidence < 0.7:
                report.warnings.append(f"{field_key}: Low confidence ({field_conf.confidence:.2f})")

        # Schema completeness: how many expected fields were extracted
        extracted_keys = set(extracted_result.keys())
        if expected_fields:
            report.schema_completeness = len(extracted_keys & expected_fields) / len(expected_fields)
        else:
            report.schema_completeness = 1.0 if extracted_keys else 0.0

        # Overall confidence: weighted average
        if report.field_confidences:
            confidence_values = [fc.confidence for fc in report.field_confidences.values()]
            avg_confidence = sum(confidence_values) / len(confidence_values)
            # Weight: 70% field confidence, 30% schema completeness
            report.overall_confidence = avg_confidence * 0.7 + report.schema_completeness * 0.3
        else:
            report.overall_confidence = 0.0

        return report.to_dict()

    def _calculate_field_confidence(
        self,
        field_key: str,
        field_data: Any,
        signal: dict[str, Any],
    ) -> FieldConfidence:
        """
        Calculate confidence for a single field.

        Confidence = source_confidence * 0.3 + completeness * 0.3 + value_quality * 0.4
        """
        conf = FieldConfidence(field_key=field_key)

        # 1. Source confidence (from extraction signals)
        conf.source_confidence = self._calculate_source_confidence(signal)

        # 2. Completeness and value quality depend on data type and source
        source_type = signal.get("source", "")

        if source_type == "content_regex":
            # Content regex extracts single value: {"key_name": "value"}
            # Completeness = 1.0 if matched, value quality based on the value
            conf.completeness = 1.0 if signal.get("matched") else 0.0
            if isinstance(field_data, dict) and field_data:
                value = next(iter(field_data.values()), "")
                conf.value_quality = 1.0 if value and str(value).strip() else 0.0
            else:
                conf.value_quality = 0.0

        elif isinstance(field_data, dict):
            # Object type (policy_holder, insured, etc.)
            conf.completeness = self._calculate_object_completeness(field_key, field_data)
            conf.value_quality, warnings = self._calculate_object_value_quality(field_data)
            conf.warnings.extend(warnings)

        elif isinstance(field_data, list):
            # List type (coverage, cvg_premium, etc.)
            conf.completeness = self._calculate_list_completeness(field_data)
            conf.value_quality, warnings = self._calculate_list_value_quality(field_data)
            conf.warnings.extend(warnings)

        else:
            # Simple type
            conf.completeness = 1.0 if field_data else 0.0
            conf.value_quality = 1.0 if field_data and str(field_data).strip() else 0.0

        # Combined confidence with weights
        conf.confidence = conf.source_confidence * 0.3 + conf.completeness * 0.3 + conf.value_quality * 0.4

        return conf

    def _calculate_source_confidence(self, signal: dict[str, Any]) -> float:
        """
        Calculate confidence based on extraction source.

        - table_header (exact match): 1.0
        - content_regex (matched): 0.95
        - grid_fallback (matched): 0.85
        - not matched: 0.0
        """
        if not signal:
            return 0.5  # No signal available, assume moderate confidence

        if not signal.get("matched", False):
            return 0.0

        source = signal.get("source", "")

        if source == "table_header":
            return 1.0
        elif source == "content_regex":
            return 0.95
        elif source == "grid_fallback":
            # Grid fallback with data is still reliable
            kv_count = signal.get("kv_count", 0)
            if kv_count >= 3:
                return 0.9
            elif kv_count >= 1:
                return 0.8
            else:
                return 0.6
        else:
            return 0.5

    def _calculate_object_completeness(self, field_key: str, data: dict[str, Any]) -> float:
        """
        Calculate completeness for object-type fields.
        """
        expected = self.EXPECTED_FIELDS.get(field_key, [])
        if not expected:
            # No expected fields defined, use extracted count
            return min(len(data) / 4, 1.0) if data else 0.0

        extracted_keys = set(data.keys())
        expected_keys = set(expected)
        matched = len(extracted_keys & expected_keys)
        return matched / len(expected_keys) if expected_keys else 0.0

    def _calculate_object_value_quality(self, data: dict[str, Any]) -> tuple[float, list[str]]:
        """
        Calculate value quality for object-type fields.
        Returns (quality_score, warnings)
        """
        if not data:
            return 0.0, ["Empty object"]

        warnings = []
        valid_count = 0
        total_count = len(data)

        for key, value in data.items():
            if not value or not str(value).strip():
                warnings.append(f"Empty value for '{key}'")
                continue

            # Validate specific field types
            validator_type = self.FIELD_VALIDATORS.get(key)
            if validator_type:
                is_valid, error = self._validate_value(value, validator_type)
                if not is_valid:
                    warnings.append(f"Invalid {key}: {error}")
                    valid_count += 0.5  # Partial credit for having a value
                else:
                    valid_count += 1
            else:
                valid_count += 1

        return valid_count / total_count if total_count > 0 else 0.0, warnings

    def _calculate_list_completeness(self, data: list) -> float:
        """
        Calculate completeness for list-type fields.
        """
        if not data:
            return 0.0

        # At least one item with multiple fields is good
        if len(data) >= 1 and isinstance(data[0], dict) and len(data[0]) >= 2:
            return 1.0
        elif len(data) >= 1:
            return 0.7
        return 0.0

    def _calculate_list_value_quality(self, data: list) -> tuple[float, list[str]]:
        """
        Calculate value quality for list-type fields.
        Returns (quality_score, warnings)
        """
        if not data:
            return 0.0, ["Empty list"]

        warnings = []
        total_quality = 0.0

        for idx, item in enumerate(data):
            if isinstance(item, dict):
                item_quality, item_warnings = self._calculate_object_value_quality(item)
                total_quality += item_quality
                warnings.extend([f"Item {idx}: {w}" for w in item_warnings])
            elif item:
                total_quality += 1.0
            else:
                warnings.append(f"Item {idx}: Empty")

        avg_quality = total_quality / len(data) if data else 0.0
        return avg_quality, warnings

    def _validate_value(self, value: Any, validator_type: str) -> tuple[bool, Optional[str]]:
        """
        Validate a value against its expected type.
        Returns (is_valid, error_message)
        """
        value_str = str(value).strip()

        if validator_type == "date":
            date_patterns = [
                r"\d{4}-\d{2}-\d{2}",
                r"\d{4}年\d{2}月\d{2}日",
                r"\d{4}/\d{2}/\d{2}",
            ]
            if any(re.search(p, value_str) for p in date_patterns):
                return True, None
            return False, f"Invalid date format: {value_str}"

        elif validator_type == "gender":
            if value_str in ["男", "女", "M", "F", "Male", "Female"]:
                return True, None
            return False, f"Invalid gender: {value_str}"

        elif validator_type == "id_number":
            # Chinese ID: 18 digits, last may be X
            if re.match(r"^[0-9A-Z]{10,18}$", value_str):
                return True, None
            return False, f"Invalid ID number format: {value_str}"

        elif validator_type == "phone":
            if re.match(r"^1[3-9]\d{9}$", value_str):
                return True, None
            return False, f"Invalid phone format: {value_str}"

        elif validator_type == "email":
            if re.match(r"^[\w\.-]+@[\w\.-]+\.\w+$", value_str):
                return True, None
            return False, f"Invalid email format: {value_str}"

        return True, None


def print_confidence_report(report: dict):
    """
    Print confidence report in readable format.
    """
    print("=" * 60)
    print("Data Extraction Confidence Report")
    print("=" * 60)
    print(f"\nOverall Confidence: {report['overall_confidence']:.2%}")
    print(f"Schema Completeness: {report['schema_completeness']:.2%}")

    print("\nField Confidence Details:")
    for field_key, field_info in report.get("field_confidence", {}).items():
        conf = field_info["confidence"]
        status = "✓" if conf >= 0.9 else "⚠" if conf >= 0.7 else "✗"
        print(f"\n  {status} {field_key}: {conf:.2%}")
        print(f"      Source: {field_info['source_confidence']:.2%}")
        print(f"      Completeness: {field_info['completeness']:.2%}")
        print(f"      Value Quality: {field_info['value_quality']:.2%}")

        if field_info.get("warnings"):
            for warning in field_info["warnings"]:
                print(f"      ⚠ {warning}")

    if report.get("warnings"):
        print(f"\n⚠ Warnings ({len(report['warnings'])}):")
        for warning in report["warnings"]:
            print(f"  - {warning}")

    if report.get("errors"):
        print(f"\n✗ Errors ({len(report['errors'])}):")
        for error in report["errors"]:
            print(f"  - {error}")

    print("\n" + "=" * 60)
