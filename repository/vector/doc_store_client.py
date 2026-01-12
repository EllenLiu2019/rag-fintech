from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class DocStoreClient(ABC):
    """
    Database operations
    """

    @abstractmethod
    def dbType(self) -> str:
        """
        Return the type of the database.
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def health(self) -> dict:
        """
        Return the health status of the database.
        """
        raise NotImplementedError("Not implemented")

    """
    Table operations
    """

    @abstractmethod
    def createIdx(self, knowledgebaseId: str, vectorSize: int):
        """
        Create an index with given name
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def deleteIdx(self, knowledgebaseId: str):
        """
        Delete an index with given name
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def indexExist(self, knowledgebaseId: str) -> bool:
        """
        Check if an index with given name exists
        """
        raise NotImplementedError("Not implemented")

    """
    CRUD operations
    """

    @abstractmethod
    def search(
        self,
        selectFields: list[str],
        query_vectors: List[List[float]],
        limit: int,
        knowledgebaseIds: list[str],
        filters: Optional[Dict | str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search with given filters and return all fields of matched documents
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get(self, chunkId: str, knowledgebaseIds: list[str]) -> dict | None:
        """
        Get single chunk with given id
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def insert(self, rows: list[dict], knowledgebaseId: str = None) -> list[str]:
        """
        Update or insert a bulk of rows
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update(self, condition: dict, newValue: dict, knowledgebaseId: str) -> bool:
        """
        Update rows with given conjunctive equivalent filtering condition
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete(self, condition: dict, knowledgebaseId: str) -> int:
        """
        Delete rows with given conjunctive equivalent filtering condition
        """
        raise NotImplementedError("Not implemented")


# Helper function to safely extract field values
def _safe_str(value, default=""):
    """Convert value to string, handling None."""
    return str(value) if value is not None else default


def extract_entity_fields(entity_dict: dict, selectFields: list[str], field_schema: dict = None) -> dict:
    """
    Extract entity fields from entity dict, preserving types based on schema.

    Args:
        entity_dict: Dictionary containing entity data from Milvus
        selectFields: List of fields to extract
        field_schema: Optional schema definition for fields (from milvus_mapping["fields"])
                     If provided, uses schema to determine type preservation

    Note: entity_dict is guaranteed to be a dict type because:
    1. Hit class initializes entity as {} (empty dict)
    2. Hit inherits from UserDict, and hit["entity"] accesses self.data["entity"]
    3. All field data is populated using dict methods (__setitem__)
    """
    result = {}
    for field in selectFields:
        if field in entity_dict:
            value = entity_dict.get(field)

            # Determine if we should preserve the type
            preserve_type = False
            if field_schema and field in field_schema:
                data_type = field_schema[field].get("data_type", "").lower()
                # Preserve numeric types: int64, int32, float, double
                preserve_type = data_type in ("int64", "int32", "float", "double")
            else:
                # Fallback: preserve if value is numeric
                preserve_type = isinstance(value, (int, float))

            # Keep JSON fields (like business_data) as their original type (dict/list)
            if field == "business_data" and isinstance(value, (dict, list)):
                result[field] = value
            # Preserve numeric types based on schema or value type
            elif preserve_type and isinstance(value, (int, float)):
                result[field] = value
            else:
                result[field] = _safe_str(value)
    return result
