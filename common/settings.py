import os
import re

from common.config_utils import get_base_config

import logging

logger = logging.getLogger(__name__)

VECTOR_STORE = None
RDB_CLIENT = None
MILVUS = {}
RDB = {}
# ${VAR:-default}
SETTING_PATTERN = r"\$\{(\w+)(?::(-[^}]*))?\}"


def init_settings():

    global VECTOR_STORE, MILVUS, RDB_CLIENT, RDB
    DOC_ENGINE = os.environ.get("DOC_ENGINE", "milvus")
    RDB_TYPE = os.getenv("DB_TYPE", "postgresql")

    lower_case_doc_engine = DOC_ENGINE.lower()
    lower_case_rdb_type = RDB_TYPE.lower()
    if lower_case_doc_engine == "milvus":
        MILVUS = replace_env_vars(get_base_config("milvus", {"host": "http://localhost:19530", "token": "root:Milvus"}))

        from repository.vector.milvus_client import VectorStoreClient

        VECTOR_STORE = VectorStoreClient()
    if lower_case_rdb_type == "postgresql":
        RDB = replace_env_vars(get_base_config("postgresql", {}))

        from repository.rdb.postgresql_client import PostgreSQLClient

        RDB_CLIENT = PostgreSQLClient()
    else:
        raise Exception(f"Not supported vector store engine: {DOC_ENGINE}")


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
