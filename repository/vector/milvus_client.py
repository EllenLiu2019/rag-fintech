from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    model,
    AnnSearchRequest,
)
from common import config
from common.decorator import singleton
from .doc_store_client import DocStoreClient, extract_entity_fields
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


@singleton
class VectorStoreClient(DocStoreClient):
    def __init__(self):

        milvus_config = config.MILVUS
        self.uri = milvus_config.get("host", "http://localhost:19530")
        self.token = milvus_config.get("token", "root:Milvus")

        logger.info(f"Use Milvus at {self.uri} as the doc engine.")

        try:
            self.client = MilvusClient(uri=self.uri, token=self.token)
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

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=True,
            description=f"RAGFlow collection for {collection_name}",
        )

        for field_name, field_config in config.MILVUS_MAPPING["fields"].items():
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
        for field_name, index_config in config.MILVUS_MAPPING["indexes"].items():
            if field_name == "dense_vector":
                field_name = f"{field_name}_{vectorSize}"
                index_config["index_name"] = f"{index_config['index_name']}_{vectorSize}"
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
        query_vector: list[float],
        limit: int,
        indexNames: str | list[str],
        knowledgebaseIds: list[str],
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")

        collection_name = f"{indexNames[0]}_{knowledgebaseIds[0]}"

        filter_expr = self._build_filter_expr(filters)

        search_res = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field=f"dense_vector_{len(query_vector)}",
            filter=filter_expr,
            limit=limit,
            output_fields=selectFields,
            search_params={"metric_type": "COSINE"},
        )

        res = [
            {
                "id": str(hit.id),
                "distance": hit.distance,
                "score": hit.score,
                **extract_entity_fields(hit["entity"], selectFields),
            }
            for hits in search_res
            for hit in hits
        ]
        return res

    def hybrid_search(
        self,
        selectFields: list[str],
        query: str,
        query_vector: list[float],
        limit: int,
        indexNames: str | list[str],
        knowledgebaseIds: list[str],
        filters: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        if isinstance(indexNames, str):
            indexNames = indexNames.split(",")

        collection_name = f"{indexNames[0]}_{knowledgebaseIds[0]}"

        filter_expr = self._build_filter_expr(filters)

        # Handle vector search parameters
        dense_search_params = {
            "data": [query_vector],
            "anns_field": f"dense_vector_{len(query_vector)}",
            "param": {"nprobe": 10},
            "limit": limit,
            "expr": filter_expr,
        }
        dense_request = AnnSearchRequest(**dense_search_params)

        sparse_search_params = {
            "data": self.bge_m3_embedding_function([query])["sparse"],
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

        search_res = self.client.hybrid_search(
            collection_name=collection_name,
            reqs=requests,
            ranker=ranker,
            filters=filter_expr,
            limit=limit,
            output_fields=selectFields,
        )

        res = [
            {
                "id": str(hit.id),
                "distance": hit.distance,
                "score": hit.score,
                **extract_entity_fields(hit["entity"], selectFields),
            }
            for hits in search_res
            for hit in hits
        ]
        return res

    def insert(self, chunks: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        collection_name = f"{indexName}_{knowledgebaseId}"

        vector_size = 0
        if chunks and "dense_vector" in chunks[0]:
            vector_size = len(chunks[0]["dense_vector"])
        if vector_size == 0:
            logger.exception(f"Cannot create collection {collection_name}: unknown vector size.")
            raise ValueError(f"{collection_name}: unknown vector size.")

        if not self.indexExist(indexName, knowledgebaseId):
            self.createIdx(indexName, knowledgebaseId, vector_size)

        data_to_insert = []
        field_names = config.MILVUS_MAPPING["fields"].keys()

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
