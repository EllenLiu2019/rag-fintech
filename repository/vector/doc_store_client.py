from abc import ABC, abstractmethod
import numpy as np


DEFAULT_MATCH_VECTOR_TOPN = 5
VEC = list | np.ndarray


class MatchTextExpr(ABC):
    def __init__(
        self,
        fields: list[str],
        matching_text: str,
        topn: int,
        extra_options: dict = dict(),
    ):
        self.fields = fields
        self.matching_text = matching_text
        self.topn = topn
        self.extra_options = extra_options


class MatchDenseExpr(ABC):
    def __init__(
        self,
        vector_column_name: str,
        embedding_data: VEC,
        embedding_data_type: str,
        distance_type: str,
        topn: int = DEFAULT_MATCH_VECTOR_TOPN,
        extra_options: dict = dict(),
    ):
        self.vector_column_name = vector_column_name
        self.embedding_data = embedding_data
        self.embedding_data_type = embedding_data_type
        self.distance_type = distance_type
        self.topn = topn
        self.extra_options = extra_options


MatchExpr = MatchDenseExpr | MatchTextExpr


class OrderByExpr(ABC):
    def __init__(self):
        self.fields = list()

    def asc(self, field: str):
        self.fields.append((field, 0))
        return self

    def desc(self, field: str):
        self.fields.append((field, 1))
        return self

    def fields(self):
        return self.fields


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
    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):
        """
        Create an index with given name
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        """
        Delete an index with given name
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def indexExist(self, indexName: str, knowledgebaseId: str) -> bool:
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
        matchExprs: list[MatchExpr],
        selectFields: list[str],
        limit: int,
        condition: dict,
        knowledgebaseIds: list[str],
        indexNames: str | list[str],
    ):
        """
        Search with given conjunctive equivalent filtering condition and return all fields of matched documents
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def get(self, chunkId: str, indexName: str, knowledgebaseIds: list[str]) -> dict | None:
        """
        Get single chunk with given id
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def insert(self, rows: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        """
        Update or insert a bulk of rows
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update(self, condition: dict, newValue: dict, indexName: str, knowledgebaseId: str) -> bool:
        """
        Update rows with given conjunctive equivalent filtering condition
        """
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        """
        Delete rows with given conjunctive equivalent filtering condition
        """
        raise NotImplementedError("Not implemented")


# Helper function to safely extract field values
def _safe_str(value, default=""):
    """Convert value to string, handling None."""
    return str(value) if value is not None else default


def extract_entity_fields(entity_dict: dict, selectFields: list[str]) -> dict:
    """
    Extract entity fields from entity dict.

    Note: entity_dict is guaranteed to be a dict type because:
    1. Hit class initializes entity as {} (empty dict)
    2. Hit inherits from UserDict, and hit["entity"] accesses self.data["entity"]
    3. All field data is populated using dict methods (__setitem__)
    """
    # entity_dict is always a dict (from pymilvus Hit class)
    result = {}
    for field in selectFields:
        if field in entity_dict:
            value = entity_dict.get(field)
            # Keep JSON fields (like business_data) as their original type (dict/list)
            # Only convert other fields to string
            if field == "business_data" and isinstance(value, (dict, list)):
                result[field] = value
            else:
                result[field] = _safe_str(value)
    return result
