from typing import List, Dict, Any, Optional
from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    model,
    AnnSearchRequest,
    MilvusException,
)

from common import get_logger, file_utils
from common.decorator import singleton
from common.config_utils import get_base_config, load_yaml_conf
from common.constants import MILVUS_MAPPING_CONF, VECTOR_STORE_NAME
from repository.vector.doc_store_client import DocStoreClient, extract_entity_fields
from common.exceptions import ConnectionError, VectorStoreError
from common.error_codes import ErrorCodes
from repository.cache import cached

logger = get_logger(__name__)


@singleton
class VectorStoreClient(DocStoreClient):

    def __init__(self, config: dict = None):
        # Support both DI and standalone usage
        milvus_config = config or get_base_config("milvus", {})
        self.uri = milvus_config.get("host", "http://localhost:19530")
        self.token = milvus_config.get("token", "root:Milvus")
        self.milvus_mapping = load_yaml_conf(file_utils.get_project_root_dir("conf", MILVUS_MAPPING_CONF))

        logger.info(f"Use Milvus at {self.uri} as the doc engine.")

        try:
            self._init_client()
            logger.info("Milvus client initialized successfully.")
            self.bge_m3_embedding_function = model.hybrid.BGEM3EmbeddingFunction(
                device="cpu",
                normalize_embeddings=False,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            logger.info("BGE-M3 embedding function initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Milvus client: {e}")
            raise

    def _init_client(self):
        """Create or reconnect to Milvus."""
        try:
            self._client = MilvusClient(uri=self.uri, token=self.token)
            logger.info(f"Connected to Milvus at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    @property
    def client(self) -> MilvusClient:
        """Get client with auto-reconnect on failure."""
        if self._client is None:
            self._init_client()
        return self._client

    def dbType(self) -> str:
        return "milvus"

    def health(self) -> dict:
        try:
            self.client.list_collections()
            return {"type": "milvus", "status": "green", "error": ""}
        except Exception as e:
            return {"type": "milvus", "status": "red", "error": str(e)}

    def createIdx(self, knowledgebaseId: str, vectorSize: int):

        collection_name = f"{VECTOR_STORE_NAME}_{knowledgebaseId}"

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
                "default": field_config.get("default", None),
            }

            if "max_length" in field_config:
                field_kwargs["max_length"] = field_config["max_length"]

            if field_config.get("enable_analyzer", False):
                field_kwargs["enable_analyzer"] = True
                bm25_function = Function(
                    name=f"{field_name}_bm25_emb",
                    input_field_names=[field_name],
                    output_field_names=["sparse_vector"],
                    function_type=FunctionType.BM25,
                )
                schema.add_function(bm25_function)

            if "dim" in field_config:
                field_kwargs["field_name"] = f"{field_name}_{vectorSize}"
                field_kwargs["dim"] = vectorSize

            schema.add_field(**field_kwargs)

        index_params = self.client.prepare_index_params()
        for field_name, index_config in self.milvus_mapping["indexes"].items():
            index_config_copy = index_config.copy()

            if field_name == "dense_vector":
                index_config_copy["field_name"] = f"{field_name}_{vectorSize}"
                index_config_copy["index_name"] = f"{index_config_copy['index_name']}_{vectorSize}"
            else:
                index_config_copy["field_name"] = field_name

            index_params.add_index(**index_config_copy)

        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
        )

        logger.info(f"Milvus created collection {collection_name}, vector size {vectorSize}")

    def deleteIdx(self, knowledgebaseId: str):
        collection_name = f"{VECTOR_STORE_NAME}_{knowledgebaseId}"
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            logger.info(f"Milvus dropped collection {collection_name}")

    def indexExist(self, knowledgebaseId: str) -> bool:
        collection_name = f"{VECTOR_STORE_NAME}_{knowledgebaseId}"
        return self.client.has_collection(collection_name)

    def search(
        self,
        selectFields: list[str],
        query_vectors: List[List[float]],
        limit: int,
        knowledgebaseIds: list[str],
        filters: Optional[Dict | str] = None,
    ) -> List[Dict[str, Any]]:
        collection_name = f"{VECTOR_STORE_NAME}_{knowledgebaseIds[0]}"

        if isinstance(filters, dict):
            filter_expr = self._build_filter_expr(filters)
        elif isinstance(filters, str):
            filter_expr = filters
        else:
            raise ValueError(f"Invalid filters type: {type(filters)}")

        try:
            search_res = self.client.search(
                collection_name=collection_name,
                data=query_vectors,
                anns_field=(
                    f"dense_vector_{len(query_vectors[0])}" if knowledgebaseIds[0] == "default_kb" else "dense_vector"
                ),
                filter=filter_expr,
                limit=limit,
                output_fields=selectFields,
                search_params={
                    "metric_type": "COSINE",
                    "params": {"ef": min(limit * 2, 100)},
                },
            )

        except MilvusException as e:
            if "connection" in str(e).lower():
                raise ConnectionError(
                    message="Failed to connect to Milvus",
                    code=ErrorCodes.R_VECTOR_001,
                    details={"uri": self.uri},
                )
            raise VectorStoreError(
                message="Milvus search failed",
                code=ErrorCodes.R_VECTOR_002,
                details={"collection": collection_name, "error": str(e)},
            )

        results = []
        for idx, hits in enumerate(search_res):
            results.append([])
            for hit in hits:
                results[idx].append(
                    {
                        "id": str(hit.id),
                        "distance": hit.distance,
                        "score": hit.score,
                        **extract_entity_fields(hit["entity"], selectFields),
                    }
                )

        return results

    def hybrid_search(
        self,
        selectFields: list[str],
        optimized_queries: List[str],
        query_vectors: List[List[float]],
        limit: int,
        knowledgebaseIds: list[str],
        filters: Optional[Dict | str] = None,
    ) -> List[Dict[str, Any]]:
        collection_name = f"{VECTOR_STORE_NAME}_{knowledgebaseIds[0]}"

        if isinstance(filters, dict):
            filter_expr = self._build_filter_expr(filters)
        elif isinstance(filters, str):
            filter_expr = filters
        else:
            raise ValueError(f"Invalid filters type: {type(filters)}")

        # Handle vector search parameters
        dense_search_params = {
            "data": query_vectors,
            "anns_field": (
                f"dense_vector_{len(query_vectors[0])}" if knowledgebaseIds[0] == "default_kb" else "dense_vector"
            ),
            # Optimize ef for serverless: lower ef = faster queries
            "param": {"ef": min(limit * 2, 100)},
            "limit": limit,
            "expr": filter_expr,
        }
        dense_request = AnnSearchRequest(**dense_search_params)

        sparse_search_params = {
            "data": self._embed_query(optimized_queries),
            "anns_field": "sparse_vector",
            "param": {"drop_ratio_search": 0.2},
            "limit": limit,
            "expr": filter_expr,
        }
        sparse_request = AnnSearchRequest(**sparse_search_params)
        requests = [dense_request, sparse_request]

        ranker = Function(
            name="rrf",
            input_field_names=[],  # Must be an empty list
            function_type=FunctionType.RERANK,
            params={"reranker": "rrf", "k": 60},
        )

        try:
            search_res = self.client.hybrid_search(
                collection_name=collection_name,
                reqs=requests,
                ranker=ranker,
                filters=filter_expr,
                limit=limit,
                output_fields=selectFields,
            )
        except MilvusException as e:
            if "connection" in str(e).lower():
                raise ConnectionError(
                    message="Failed to connect to Milvus",
                    code=ErrorCodes.R_VECTOR_001,
                    details={"uri": self.uri},
                )
            raise VectorStoreError(
                message="Milvus search failed",
                code=ErrorCodes.R_VECTOR_002,
                details={"collection": collection_name, "error": str(e)},
            )

        results = []
        for idx, hits in enumerate(search_res):
            results.append([])
            for hit in hits:
                results[idx].append(
                    {
                        "id": str(hit.id),
                        "distance": hit.distance,
                        "score": hit.score,
                        **extract_entity_fields(hit["entity"], selectFields),
                    }
                )

        return results

    def insert(self, chunks: list[dict], knowledgebaseId: str = None) -> list[str]:
        collection_name = f"{VECTOR_STORE_NAME}_{knowledgebaseId}"

        vector_size = 0
        if chunks and "dense_vector" in chunks[0]:
            vector_size = len(chunks[0]["dense_vector"])
        if vector_size == 0:
            logger.exception(f"Cannot create collection {collection_name}: unknown vector size.")
            raise ValueError(f"{collection_name}: unknown vector size.")

        if not self.indexExist(knowledgebaseId):
            self.createIdx(knowledgebaseId, vector_size)

        data_to_insert = []
        field_names = self.milvus_mapping["fields"].keys()

        embedding_to_use = [chunk.get("text", "") for chunk in chunks]
        sparse_vectors = self.bge_m3_embedding_function(embedding_to_use)

        for chunk, sparse_vector in zip(chunks, sparse_vectors["sparse"]):
            # Convert scipy sparse matrix to Milvus dict format
            if hasattr(sparse_vector, "tocoo"):
                # Convert CSR to COO format for easier iteration
                coo_matrix = sparse_vector.tocoo()
                sparse_dict = {int(idx): float(val) for idx, val in zip(coo_matrix.col, coo_matrix.data)}
                chunk["sparse_vector"] = sparse_dict
                logger.info(f"Converted sparse vector with {len(sparse_dict)} non-zero elements")
            else:
                chunk["sparse_vector"] = sparse_vector
                logger.warning(f"Sparse vector is not in expected scipy format: {type(sparse_vector)}")

        for chunk in chunks:
            item = {}
            for field in field_names:
                if field in chunk:
                    item[field] = chunk[field]

            if "dense_vector" in item:
                item[f"dense_vector_{vector_size}"] = chunk["dense_vector"]
                item.pop("dense_vector")

            data_to_insert.append(item)

        res = []
        for i in range(0, len(data_to_insert), 50):
            data_batch = data_to_insert[i : i + 50]
            res_batch = self.client.insert(collection_name=collection_name, data=data_batch)

            insert_count = res_batch["insert_count"]
            ids = res_batch["ids"]
            cost = res_batch["cost"]
            logger.info(f"Milvus inserted {insert_count} rows into {collection_name} with cost {cost}")
            res.extend(ids)

        return res

    def update(self, condition: dict, newValue: dict, knowledgebaseId: str) -> bool:
        collection_name = f"{VECTOR_STORE_NAME}_{knowledgebaseId}"
        self.client.upsert(collection_name=collection_name, data=newValue)
        return True

    def delete(self, condition: dict, knowledgebaseId: str) -> int:
        collection_name = f"{VECTOR_STORE_NAME}_{knowledgebaseId}"
        filter_expr = self._build_delete_expr(condition)
        res = self.client.delete(collection_name=collection_name, filter=filter_expr)
        return res

    def get(self, chunkId: str, knowledgebaseIds: list[str]) -> dict | None:
        collection_name = f"{VECTOR_STORE_NAME}_{knowledgebaseIds[0]}"
        search_res = self.client.get(collection_name=collection_name, ids=[chunkId], output_fields=["text"])
        return search_res[0]["text"] if search_res else None

    def _build_delete_expr(self, condition: dict) -> str:
        parts = []
        for k, v in condition.items():
            if isinstance(v, list):
                v_str = ", ".join([f'"{item}"' for item in v])
                parts.append(f"{k} in [{v_str}]")
            else:
                parts.append(f'{k} == "{v}"')
        return " && ".join(parts)

    def _build_filter_expr(self, filters: dict) -> str:
        parts = []
        for k, v in filters.items():
            if isinstance(v, list):
                v_str = ", ".join([f'"{item}"' for item in v])
                parts.append(f"{k} in [{v_str}]")
            else:
                parts.append(f'{k} == "{v}"')
        return " and ".join(parts) if parts else ""

    @cached(prefix="embed_sparse", ttl=1800)
    def _embed_query(self, queries: List[str]) -> List[List[float]]:
        """Generate sparse vectors for queries with performance monitoring."""
        import time

        start_time = time.time()

        try:
            result = self.bge_m3_embedding_function(queries)["sparse"]
            elapsed = time.time() - start_time
            logger.info(f"BGE-M3 sparse embedding generated in {elapsed:.3f}s for {len(queries)} query(ies)")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"BGE-M3 sparse embedding failed after {elapsed:.3f}s: {e}")
            raise
