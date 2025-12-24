from typing import Any

from common import get_logger

logger = get_logger(__name__)


def check_missing_fields(
    schema: dict[str, Any],
    extracted_entities: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    missing_fields = {}
    schema_fields = schema.get("fields", {})
    for field_key, field_config in schema_fields.items():
        db_mapping = field_config.get("db_mapping", {})
        columns = db_mapping.get("columns", {})
        if field_key not in extracted_entities:
            missing_fields[field_key] = {}
            for schema_key, prop_config in columns.items():
                missing_fields[field_key][schema_key] = ""
            continue

        results = extracted_entities[field_key] or {}
        if isinstance(results, dict):
            for schema_key, prop_config in columns.items():
                if schema_key not in results:
                    if field_key not in missing_fields:
                        missing_fields[field_key] = {}
                    missing_fields[field_key][schema_key] = ""
        elif isinstance(results, list):
            for result in results:
                for schema_key, prop_config in columns.items():
                    if schema_key not in result:
                        if field_key not in missing_fields:
                            missing_fields[field_key] = {}
                        missing_fields[field_key][schema_key] = ""

    logger.info(f"Missing fields: {missing_fields}")

    return missing_fields
