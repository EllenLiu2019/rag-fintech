from datetime import datetime
from decimal import Decimal
from typing import Any, Optional, Type

from repository.rdb.models import PolicyHolder, Insured, Coverage, CvgPremium, Policy, Base
from repository.rdb.postgresql_client import PostgreSQLClient
from common import get_logger

logger = get_logger(__name__)


SCHEMA_TABLE_MAP = {
    "policy": Policy,
    "policy_holder": PolicyHolder,
    "insured": Insured,
    "coverage": Coverage,
    "cvg_premium": CvgPremium,
}


class RdbBuilder:
    def __init__(
        self,
        schema: dict[str, Any],
        extracted_results: dict[str, dict[str, Any]],
        confidence_score: float = 0.0,
        source_file: str = "unknown",
    ) -> None:
        self.schema = schema
        self.extracted_results = extracted_results
        self.confidence_score = confidence_score
        self.source_file = source_file
        self.rdb_client = PostgreSQLClient()
        self.extracted_entities = self._build_entities()

    def build(self) -> None:
        """
        Build and save all entities to database.
        """
        try:
            policy_number = self.check_policy_exists()
            if policy_number is None:
                logger.warning("Policy number not found, cannot proceed")
                return

            for table, model in SCHEMA_TABLE_MAP.items():
                self._build_table(table, model)

            logger.info(f"Successfully saved policy {policy_number} to database")

        except Exception as e:
            logger.error(f"Failed to build and save entities to database: {e}", exc_info=True)

    def check_policy_exists(self) -> Optional[str]:
        policy_data = self.extracted_entities.get("policy", {})
        if not policy_data:
            logger.warning("Policy data not found")
            return None

        policy_number = None
        if "policy_number" in policy_data:
            policy_number_tuple = policy_data["policy_number"]
            if policy_number_tuple:
                policy_number, _ = policy_number_tuple

        if not policy_number:
            logger.warning("Policy number not found, cannot proceed")
            return None

        existing_policy = self.rdb_client.select_by_kwargs(Policy, policy_number=policy_number)

        if existing_policy:
            logger.info(f"Policy {policy_number} already exists, updating all related records to status 'I'")
            self._inactivate_existing_policy(policy_number)

        return policy_number

    def _build_entities(self) -> dict[str, dict[str, Any]]:
        """
        Build entities from extracted result for database storage.
        """
        extracted_entities = {}

        for key, results in self.extracted_results.items():
            if key not in self.schema.get("fields", {}).keys():
                continue

            field_config = self.schema.get("fields", {}).get(key, {})
            db_mapping = field_config.get("db_mapping", {})
            tables = db_mapping.get("tables", [])
            columns = db_mapping.get("columns", {})

            if isinstance(results, dict):
                entity = {
                    prop_config.get("name"): (results[schema_key], prop_config.get("type"))
                    for schema_key, prop_config in columns.items()
                    if schema_key in results
                }
                for table in tables:
                    if table not in extracted_entities:
                        extracted_entities[table] = {}
                    extracted_entities[table].update(entity)
            elif isinstance(results, list):
                entities = [
                    {
                        prop_config.get("name"): (row[schema_key], prop_config.get("type"))
                        for schema_key, prop_config in columns.items()
                        if schema_key in row
                    }
                    for row in results
                ]

                for table in tables:
                    if table in extracted_entities:
                        existing_data = extracted_entities[table]
                        if isinstance(existing_data, dict):
                            for entity in entities:
                                entity.update({k: v for k, v in existing_data.items() if k not in entity})
                        elif isinstance(existing_data, list):
                            entities = existing_data + entities

                    extracted_entities[table] = entities

        logger.info(f"Extracted entities: {extracted_entities}")
        return extracted_entities

    def _inactivate_existing_policy(self, policy_number: str) -> None:
        """Update all records for a given policy_number to status 'I' (inactive)"""
        update_kwargs = {"status": "I"}
        filter_kwargs = {"policy_number": policy_number}

        for table, model in SCHEMA_TABLE_MAP.items():
            try:
                count = self.rdb_client.update_many_by_kwargs(model, filter_kwargs, update_kwargs)
                if count > 0:
                    logger.info(f"Updated {count} {table} records to status 'I' for policy {policy_number}")
            except Exception as e:
                logger.warning(f"Failed to update {table} records: {e}", exc_info=True)
                continue

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string with multiple format support"""
        if not date_str:
            return None

        # Remove common Chinese date separators
        date_str = date_str.replace("年", "-").replace("月", "-").replace("日", "").strip()

        formats = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"]
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        logger.warning(f"Failed to parse date: {date_str}")
        return None

    def _parse_decimal(self, value: Any) -> Optional[Decimal]:
        """Parse decimal value, handling strings, int, float, and other numeric types"""
        if value is None:
            return None

        # If already a Decimal, return as is
        if isinstance(value, Decimal):
            return value

        # If already a number (int or float), convert directly
        if isinstance(value, (int, float)):
            try:
                return Decimal(str(value))
            except (ValueError, Exception) as e:
                logger.warning(f"Failed to convert number to decimal: {value}, error: {e}")
                return None

        # If string, clean and parse
        if isinstance(value, str):
            if not value.strip():
                return None
            try:
                # Remove commas and other separators
                cleaned = value.replace(",", "").replace(" ", "").strip()
                return Decimal(cleaned)
            except (ValueError, Exception) as e:
                logger.warning(f"Failed to parse decimal string: {value}, error: {e}")
                return None

        # Try to convert other types
        try:
            return Decimal(str(value))
        except (ValueError, Exception) as e:
            logger.warning(f"Failed to parse decimal: {value}, error: {e}")
            return None

    def _parse_float(self, value: Any) -> Optional[float]:
        """Parse float value, handling strings, int, float, and other numeric types"""
        if value is None:
            return None

        # If already a float, return as is
        if isinstance(value, float):
            return value

        # If already an int, convert to float
        if isinstance(value, int):
            return float(value)

        # If string, clean and parse
        if isinstance(value, str):
            if not value.strip():
                return None
            try:
                # Remove commas and other separators
                cleaned = value.replace(",", "").replace(" ", "").strip()
                return float(cleaned)
            except (ValueError, Exception) as e:
                logger.warning(f"Failed to parse float string: {value}, error: {e}")
                return None

        # Try to convert other types
        try:
            return float(value)
        except (ValueError, Exception) as e:
            logger.warning(f"Failed to parse float: {value}, error: {e}")
            return None

    def _get_value(self, entity_data: dict, key: str, default: Any = None) -> Any:
        """Safely get value from entity_data"""
        if key in entity_data:
            value, _ = entity_data[key]
            return value
        return default

    def _convert_value(self, value: Any, value_type: str) -> Any:
        """Convert value based on type"""
        if value_type == "date":
            # Date values should be strings
            if isinstance(value, str):
                return self._parse_date(value)
            else:
                return value
        elif value_type == "decimal":
            return self._parse_decimal(value)
        elif value_type == "float":
            return self._parse_float(value)
        elif value_type == "string":
            return value
        else:
            return value

    def _get_update_kwargs(self, model_instance: Base, table: str) -> dict[str, Any]:
        """Get keyword arguments for updating an existing record"""
        from sqlalchemy.inspection import inspect as sa_inspect

        update_kwargs = {}

        # Handle special fields for Policy
        if table == "policy":
            update_kwargs["source_file"] = self.source_file
            update_kwargs["extraction_time"] = datetime.now()
            update_kwargs["confidence_score"] = self.confidence_score

        # Get all column attributes (excluding relationships and id)
        mapper = sa_inspect(model_instance).mapper
        for column_attr in mapper.column_attrs:
            key = column_attr.key
            if key != "id" and hasattr(model_instance, key):
                value = getattr(model_instance, key)
                # Include value even if None (to allow clearing fields)
                update_kwargs[key] = value

        return update_kwargs

    def _build_table(self, table: str, model: Type[Base]) -> None:
        """
        Build and save table entities.
        All tables use the same logic: check if exists, update if exists, otherwise save.
        """
        table_data = self.extracted_entities.get(table, {})
        if not table_data:
            logger.warning(f"No data found for table {table}")
            return

        try:
            if isinstance(table_data, dict):
                # Single entity (policy_holder, insured, policy)
                model_instance = model()

                for key, entity_tuple in table_data.items():
                    value_str, value_type = entity_tuple
                    converted_value = self._convert_value(value_str, value_type)
                    setattr(model_instance, key, converted_value)

                # Handle special fields for Policy
                if table == "policy":
                    model_instance.source_file = self.source_file
                    model_instance.extraction_time = datetime.now()
                    model_instance.confidence_score = self.confidence_score

                if hasattr(model_instance, "status"):
                    model_instance.status = "A"

                saved = self.rdb_client.save(model_instance)
                identifier = getattr(saved, "name", getattr(saved, "policy_number", "N/A"))
                logger.info(f"Saved {table}: {identifier}")

            elif isinstance(table_data, list):
                # Multiple entities
                saved_instances = []
                for idx, entity_data in enumerate(table_data):
                    try:
                        model_instance = model()

                        for key, entity_tuple in entity_data.items():
                            value_str, value_type = entity_tuple
                            converted_value = self._convert_value(value_str, value_type)
                            setattr(model_instance, key, converted_value)

                        if hasattr(model_instance, "status"):
                            model_instance.status = "A"

                        saved = self.rdb_client.save(model_instance)
                        logger.info(f"Saved {table} {idx+1}: {getattr(saved, 'policy_number', 'N/A')}")

                        saved_instances.append(saved)
                    except Exception as e:
                        logger.error(f"Failed to build {table} {idx+1}: {e}", exc_info=True)

                logger.info(f"Processed {len(saved_instances)} {table} entities")

        except Exception as e:
            logger.error(f"Failed to build {table}: {e}", exc_info=True)
