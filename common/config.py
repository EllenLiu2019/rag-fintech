from common.config_utils import get_base_config, load_yaml_conf
from common.constants import MILVUS_MAPPING_CONF
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


def init_config():

    global VECTOR_STORE, MILVUS, RDB_CLIENT, RDB, REDIS, REDIS_CLIENT, MILVUS_MAPPING

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
