import os
import re

from common.config_utils import get_base_config

import logging

logger = logging.getLogger(__name__)

VECTOR_STORE = None
RDB_CLIENT = None
REDIS_CLIENT = None
MILVUS = {}
RDB = {}
# ${VAR:-default}
SETTING_PATTERN = r"\$\{(\w+)(?::(-[^}]*))?\}"
REDIS = {}


def init_settings():

    global VECTOR_STORE, MILVUS, RDB_CLIENT, RDB, REDIS, REDIS_CLIENT

    MILVUS = replace_env_vars(get_base_config("milvus", {"host": "http://localhost:19530", "token": "root:Milvus"}))
    RDB = replace_env_vars(get_base_config("postgresql", {}))
    REDIS = replace_env_vars(get_base_config("redis", {}))

    from repository.vector.milvus_client import VectorStoreClient
    from repository.rdb.postgresql_client import PostgreSQLClient

    VECTOR_STORE = VectorStoreClient()
    RDB_CLIENT = PostgreSQLClient()

    if REDIS.get("enable", True):
        from repository.cache.redis_client import RedisClient

        REDIS_CLIENT = RedisClient()


def replace_env_vars(config):
    """Recursively replace environment variables in the configuration"""
    if isinstance(config, dict):
        return {k: replace_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [replace_env_vars(item) for item in config]
    elif isinstance(config, str):
        return re.sub(
            pattern=SETTING_PATTERN,
            repl=lambda m: os.getenv(m.group(1), m.group(2)[1:] if m.group(2) else ""),
            string=config,
        )
    return config
