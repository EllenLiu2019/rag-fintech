import json
from common.config_utils import get_base_config, load_yaml_conf
from common.constants import LLM_FACTORIES_CONF, MILVUS_MAPPING_CONF, DENSE_TAG, CHAT_TAG
from common import file_utils
import logging

logger = logging.getLogger(__name__)

VECTOR_STORE = None
RDB_CLIENT = None
REDIS_CLIENT = None
MILVUS = {}
MILVUS_MAPPING = {}
RDB = {}
REDIS = {}
LLM_FACTORIES = []
CHAT_MODELS = []
EMBEDDING_MODELS = []


def init_config():

    global VECTOR_STORE, MILVUS, RDB_CLIENT, RDB, REDIS, REDIS_CLIENT, MILVUS_MAPPING, CHAT_MODELS, EMBEDDING_MODELS

    MILVUS = get_base_config("milvus", {})
    RDB = get_base_config("postgresql", {})
    REDIS = get_base_config("redis", {})
    MILVUS_MAPPING = load_yaml_conf(file_utils.get_project_root_dir("conf", MILVUS_MAPPING_CONF))

    from repository.vector.milvus_client import VectorStoreClient
    from repository.rdb.postgresql_client import PostgreSQLClient

    VECTOR_STORE = VectorStoreClient()
    RDB_CLIENT = PostgreSQLClient()

    if REDIS.get("enable", True):
        from repository.cache.redis_client import RedisClient

        REDIS_CLIENT = RedisClient()

    global LLM_FACTORIES
    try:
        with open(file_utils.get_project_root_dir("conf", LLM_FACTORIES_CONF), "r", encoding="utf-8") as f:
            LLM_FACTORIES = json.load(f)

        for model in LLM_FACTORIES:
            if CHAT_TAG in model.get("tags", []):
                CHAT_MODELS.append(model)
            if DENSE_TAG in model.get("tags", []):
                EMBEDDING_MODELS.append(model)

        logger.info(f"Loaded {len(CHAT_MODELS)} chat models, {len(EMBEDDING_MODELS)} embedding models")
    except Exception:
        raise Exception(f"Failed to load LLM Factories: {LLM_FACTORIES_CONF}")
