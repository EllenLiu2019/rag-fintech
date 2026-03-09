SERVICE_CONF = "service_conf.yaml"
LLM_FACTORIES_CONF = "llm_factories.json"
MILVUS_MAPPING_CONF = "milvus_mapping.json"
MILVUS_GRAPH_MAPPING_CONF = "milvus_graph_mapping.json"

API_KEY_SUFFIX = "_API_KEY"
DENSE_TAG = "DENSE"
SPARSE_TAG = "SPARSE"

VECTOR_STORE_NAME = "rag_fintech"
VECTOR_DEFAULT_KB = "default_kb"
VECTOR_SNOMED_KB = "snomed_kb"
VECTOR_GRAPH_KB = "graph_kb"
VECTOR_RETRIEVE_FIELDS = ["id", "text", "clause_id", "clause_path"]
VECTOR_GET_FIELDS = ["id", "text", "clause_id", "clause_path", "dense_vector_1024"]


VECTOR_GRAPH_SIMILAR_FIELDS = ["id", "entity_name", "description", "clause_ids"]
VECTOR_GRAPH_FIELDS = [
    "id",
    "graph_type",
    "entity_name",
    "entity_type",
    "clause_ids",
    "source_id",
    "target_id",
    "source_entity",
    "target_entity",
    "rel_type",
    "description",
    "doc_id",
    "root_id",
]

ACTIVE_VALUE = "A"
INACTIVE_VALUE = "I"

TUPLE_DELIMITER = "<|>"
RECORD_DELIMITER = "##"
COMPLETION_DELIMITER = "<|COMPLETE|>"
GRAPH_FIELD_SEP = "<SEP>"
ENTITY_EXTRACTION_MAX = 0
DEFAULT_ENTITY_TYPES = ["clause", "coverage_scope", "exclusion_category", "medical_entity"]
