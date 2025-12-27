from typing import Any, Optional, Type

from repository.rdb.models import Base
from repository.rdb import rdb_client
from common import get_logger, constants, field_converter
from sqlalchemy.orm import Session

logger = get_logger(__name__)


class RdbBuilder:
    def __init__(
        self,
        schema: dict[str, Any],
        extracted_results: dict[str, dict[str, Any]],
        **kwargs: Any,
    ) -> None:
        self.schema = schema
        self.kwargs = kwargs
        self.extracted_results = extracted_results
        self.extracted_entities = self._build_entities()
        self.schema_model_map = self._discover_models()

    def _discover_models(self) -> dict[str, Type[Base]]:
        """Discover all models that inherit from Base"""
        table_map = {}
        for model_class in Base.registry._class_registry.values():
            if hasattr(model_class, "__tablename__"):
                table_name = model_class.__tablename__
                table_map[table_name] = model_class
        return table_map

    def build(self) -> None:
        """
        Build and save all entities to database.
        """
        session = None
        try:
            session = rdb_client.begin_transaction()
            session.begin()

            primary_key = self.check_entity_rules(session)
            if primary_key is None:
                logger.warning("Primary key not found, cannot proceed")
                if session:
                    session.rollback()
                return

            entities = self.schema.get("entity_rules", {}).get("entities", {})
            for table, model in self.schema_model_map.items():
                self._build_table(session, table, model, entities)

            session.commit()
            logger.info(f"Successfully saved {primary_key} to database")

        except Exception as e:
            logger.error(
                f"Failed to build and save entities to database: {type(e).__name__}: {str(e)}, rolling back session"
            )
            if session:
                session.rollback()
                logger.info("Session rolled back successfully")
        finally:
            if session:
                session.close()

    def check_entity_rules(self, session: Session) -> Optional[str]:
        primary_key = None
        filter_kwargs = {}
        entity_rules = self.schema.get("entity_rules", {})
        entities = entity_rules.get("entities", {})
        for entity_name, entity in entities.items():
            if entity_name not in self.extracted_entities:
                logger.warning(f"Entity {entity_name} not found, cannot proceed")
                return None

            entity_data = self.extracted_entities.get(entity_name, {})
            primary_key = entity.get("primary_key")
            filter_kwargs.update({primary_key: entity_data.get(primary_key)[0]})

        version_control = entity_rules.get("version_control", {})
        field = version_control.get("field")
        filter_kwargs.update({field: constants.ACTIVE_VALUE})
        update_kwargs = {field: constants.INACTIVE_VALUE}

        for table, model in self.schema_model_map.items():
            try:
                count = rdb_client.update_many_by_kwargs(session, model, filter_kwargs, update_kwargs)
                if count > 0:
                    logger.info(f"Updated {count} {table} records to status 'I' for primary key {primary_key}")
            except Exception as e:
                logger.warning(f"Failed to update {table}, caused by {type(e).__name__}: {str(e)}")
                continue

        return primary_key

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
                    existing_data = extracted_entities[table]
                    if isinstance(existing_data, dict):
                        existing_data.update(entity)
                        logger.debug(f"Updated table {table} with entity: {entity}")
                    elif isinstance(existing_data, list):
                        for existing_row in existing_data:
                            existing_row.update({k: v for k, v in entity.items() if k not in existing_row})
                            logger.debug(f"Updated row {existing_row} with entity: {entity}")
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

                                logger.debug(f"Updated table {table} with entity: {entity}")
                        elif isinstance(existing_data, list):
                            entities = existing_data + entities
                            logger.debug(
                                f"Updated table {table} with existing entities: {existing_data} and new entities: {entities}"
                            )
                    extracted_entities[table] = entities
                    logger.debug(f"Updated table {table} with entities: {entities}")

        logger.info(f"Built entities: {extracted_entities}")
        return extracted_entities

    def _build_table(self, session: Session, table: str, model: Type[Base], entities: dict[str, Any]) -> None:
        """
        Build and save table entities.
        All tables use the same logic: check if exists, update if exists, otherwise save.
        """
        table_data = self.extracted_entities.get(table, {})
        if not table_data:
            logger.warning(f"No data found for table {table}")
            return

        # Get entity configuration for this table
        entity = entities.get(table, {})
        primary_key = entity.get("primary_key", "id")
        meta_fields = entity.get("meta_fields", [])

        try:
            if isinstance(table_data, dict):
                # Single entity (policy_holder, insured, policy)
                model_instance = model()

                for key, entity_tuple in table_data.items():
                    value_str, value_type = entity_tuple
                    converted_value = field_converter.convert(value_str, value_type)
                    setattr(model_instance, key, converted_value)

                # Handle meta fields
                for field in meta_fields:
                    if field in self.kwargs:
                        setattr(model_instance, field, self.kwargs[field])
                    else:
                        logger.warning(f"Meta field {field} not found in kwargs for table {table}")

                model_instance.status = constants.ACTIVE_VALUE

                saved = rdb_client.save_with_session(session, model_instance)
                identifier = getattr(saved, primary_key, "N/A")
                logger.info(f"Saved {table}: {identifier}")

            elif isinstance(table_data, list):
                # Multiple entities
                saved_instances = []
                for idx, entity_data in enumerate(table_data):
                    try:
                        model_instance = model()

                        for key, entity_tuple in entity_data.items():
                            value_str, value_type = entity_tuple
                            converted_value = field_converter.convert(value_str, value_type)
                            setattr(model_instance, key, converted_value)

                        # Handle meta fields for list entities
                        for field in meta_fields:
                            if field in self.kwargs:
                                setattr(model_instance, field, self.kwargs[field])
                            else:
                                logger.warning(f"Meta field {field} not found in kwargs for table {table}")

                        model_instance.status = constants.ACTIVE_VALUE

                        saved = rdb_client.save_with_session(session, model_instance)
                        identifier = getattr(saved, primary_key, "N/A")
                        logger.info(f"Saved {table} {idx+1}: {identifier}")

                        saved_instances.append(saved)
                    except Exception as e:
                        logger.error(f"Failed to build {table} {idx+1}: {type(e).__name__}: {str(e)}")
                        raise e

                logger.info(f"Processed {len(saved_instances)} {table} entities")

        except Exception as e:
            logger.error(f"Failed to build {table}: {type(e).__name__}: {str(e)}")
            raise e


if __name__ == "__main__":
    from pathlib import Path
    import json

    schema = Path(__file__).parent / "data" / "schema.json"
    with open(schema, "r") as f:
        schema = json.load(f)

    extracted_results = Path(__file__).parent / "data" / "extracted_results.json"
    with open(extracted_results, "r") as f:
        extracted_results = json.load(f)

    rdb_builder = RdbBuilder(
        schema=schema,
        extracted_results=extracted_results,
        confidence_score=0.95,
        source_file="test.pdf",
    )
    rdb_builder.build()
