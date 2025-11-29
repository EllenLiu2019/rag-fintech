import os
from common.config_utils import get_base_config
from repository.vector.milvus_client import VectorStoreClient
import logging

logger = logging.getLogger(__name__)

vectorStoreClient = None

MILVUS = {}


def init_settings():
    global vectorStoreClient, MILVUS
    DOC_ENGINE = os.environ.get("DOC_ENGINE", "milvus")

    lower_case_doc_engine = DOC_ENGINE.lower()
    if lower_case_doc_engine == "milvus":
        MILVUS = get_base_config("milvus", {"host": "localhost", "port": 19530})
        vectorStoreClient = VectorStoreClient()
    else:
        raise Exception(f"Not supported vector store engine: {DOC_ENGINE}")
