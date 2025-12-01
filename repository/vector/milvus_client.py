from pymilvus import MilvusClient, DataType
from common import settings
from common.config_utils import load_yaml_conf
from common.decorator import singleton
from .doc_store_client import DocStoreClient, MatchExpr, MatchDenseExpr
from common.file_utils import get_project_base_directory
import logging

logger = logging.getLogger(__name__)


@singleton
class VectorStoreClient(DocStoreClient):
    def __init__(self):
        self.milvus_mapping = load_yaml_conf(get_project_base_directory("conf", "milvus_mapping.json"))

        milvus_config = settings.MILVUS
        self.uri = f"http://{milvus_config.get('host', 'localhost')}:{milvus_config.get('port', 19530)}"
        self.token = (
            f"{milvus_config.get('user', '')}:{milvus_config.get('password', '')}" if milvus_config.get("user") else ""
        )

        logger.info(f"Use Milvus at {self.uri} as the doc engine.")

        try:
            self.client = MilvusClient(uri=self.uri, token=self.token)
            logger.info("Milvus client initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize Milvus client: {e}")
            raise

    def dbType(self) -> str:
        return "milvus"

    def health(self) -> dict:
        try:
            self.client.list_collections()
            return {"type": "milvus", "status": "green", "error": ""}
        except Exception as e:
            return {"type": "milvus", "status": "red", "error": str(e)}

    def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):

        collection_name = f"{indexName}_{knowledgebaseId}"

        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
            description=f"RAGFlow collection for {collection_name}",
        )

        for field_name, field_config in self.milvus_mapping["fields"].items():
            type_str = field_config["data_type"].upper()
            if not hasattr(DataType, type_str):
                raise ValueError(f"Unsupported Milvus DataType: {type_str}")

            field_kwargs = {
                "field_name": field_name,
                "datatype": getattr(DataType, type_str),
                "is_primary": field_config.get("is_primary", False),
                "nullable": field_config.get("nullable", False),
            }

            if "max_length" in field_config:
                field_kwargs["max_length"] = field_config["max_length"]

            if "dim" in field_config:
                field_kwargs["dim"] = vectorSize

            schema.add_field(**field_kwargs)

        index_params = self.client.prepare_index_params()
        for field_name, index_config in self.milvus_mapping["indexes"].items():
            index_params.add_index(
                field_name=field_name,
                index_name=index_config["index_name"],
                index_type=index_config["index_type"],
                metric_type=index_config["metric_type"],
                params=index_config.get("params", {}),
            )

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

        logger.info(f"Milvus created collection {collection_name}, vector size {vectorSize}")

    def deleteIdx(self, indexName: str, knowledgebaseId: str):
        collection_name = f"{indexName}_{knowledgebaseId}"
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            logger.info(f"Milvus dropped collection {collection_name}")

    def indexExist(self, indexName: str, knowledgebaseId: str) -> bool:
        collection_name = f"{indexName}_{knowledgebaseId}"
        return self.client.has_collection(collection_name)

    def search(
        self,
        selectFields: list[str],
        matchExprs: list[MatchExpr],
        limit: int,
        knowledgebaseIds: list[str],
        indexNames: str | list[str],
    ):
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")

        collection_name = f"{indexNames[0]}_{knowledgebaseIds[0]}"

        # Handle vector search parameters
        expr_parts = []
        vector_data = None
        topk = limit
        vector_field = ""
        filter_expr = ""
        for matchExpr in matchExprs:
            if isinstance(matchExpr, MatchDenseExpr):
                vector_data = matchExpr.embedding_data  # query vector
                topk = matchExpr.topn
                vector_field = matchExpr.vector_column_name
                for k, v in matchExpr.extra_options.get("filters", {}).items():
                    expr_parts.append(f'{k} == "{v}"')
                filter_expr = " and ".join(expr_parts) if expr_parts else ""
                break

        if not vector_data:
            logger.warning("No vector data provided.")
            return []

        search_res = self.client.search(
            collection_name=collection_name,
            data=[vector_data],
            anns_field=vector_field,  # dense_vector
            filter=filter_expr,
            limit=topk,
            output_fields=selectFields,
            search_params={"metric_type": "COSINE"},
        )

        # Helper function to safely extract field values
        def _safe_str(value, default=""):
            """Convert value to string, handling None."""
            return str(value) if value is not None else default

        def _extract_entity_fields(entity_dict):
            """
            Extract entity fields from entity dict.

            Note: entity_dict is guaranteed to be a dict type because:
            1. Hit class initializes entity as {} (empty dict)
            2. Hit inherits from UserDict, and hit["entity"] accesses self.data["entity"]
            3. All field data is populated using dict methods (__setitem__)
            """
            # entity_dict is always a dict (from pymilvus Hit class)
            return {
                "text": _safe_str(entity_dict.get("text")),
                "policy_number": _safe_str(entity_dict.get("policy_number")),
                "holder_name": _safe_str(entity_dict.get("holder_name")),
                "insured_name": _safe_str(entity_dict.get("insured_name")),
                "doc_id": _safe_str(entity_dict.get("doc_id")),
                "metadata": entity_dict.get("metadata", {}),
            }

        # Flatten the list of Hits (which is a list of lists) and convert to dicts
        flat_res = []
        for hits in search_res:
            for hit in hits:
                flat_res.append(
                    {
                        "id": str(hit.id),
                        "distance": hit.distance,
                        "score": hit.score,
                        **_extract_entity_fields(hit["entity"]),
                    }
                )

        return flat_res

    def insert(self, rows: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        collection_name = f"{indexName}_{knowledgebaseId}"

        if not self.indexExist(indexName, knowledgebaseId):
            vector_size = 0
            if rows and "dense_vector" in rows[0]:
                vector_size = len(rows[0]["dense_vector"])

            if vector_size > 0:
                self.createIdx(indexName, knowledgebaseId, vector_size)
            else:
                logger.warning(f"Cannot create collection {collection_name}: unknown vector size.")

        data_to_insert = []
        field_names = self.milvus_mapping["fields"].keys()

        for row in rows:
            item = {}
            for field in field_names:
                if field in row:
                    item[field] = row[field]

            data_to_insert.append(item)

        res = self.client.insert(collection_name=collection_name, data=data_to_insert)
        return res

    def update(self, condition: dict, newValue: dict, indexName: str, knowledgebaseId: str) -> bool:
        collection_name = f"{indexName}_{knowledgebaseId}"
        self.client.upsert(collection_name=collection_name, data=newValue)
        return True

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        collection_name = f"{indexName}_{knowledgebaseId}"
        filter_expr = self._build_delete_expr(condition)
        res = self.client.delete(collection_name=collection_name, filter=filter_expr)
        return res

    def get(self, chunkId: str, indexName: str, knowledgebaseIds: list[str]) -> dict | None:
        collection_name = f"{indexName}_{knowledgebaseIds[0]}"
        res = self.client.get(collection_name=collection_name, ids=[chunkId])
        return res[0] if res else None

    def _build_delete_expr(self, condition: dict) -> str:
        parts = []
        for k, v in condition.items():
            if isinstance(v, list):
                v_str = ", ".join([f'"{item}"' for item in v])
                parts.append(f"{k} in [{v_str}]")
            else:
                parts.append(f'{k} == "{v}"')
        return " && ".join(parts)
