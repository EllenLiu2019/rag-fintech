"""
Application Context - Centralized service container.
All services are explicitly initialized at module load time (eager loading).
"""

from common.log_utils import get_logger
from common.config_utils import get_base_config

logger = get_logger(__name__)


class AppContext:
    """
    Application-wide service container.
    """

    def __init__(self):

        # RAG services (these are module-level singletons)
        from rag.retrieval.reranker import reranker
        from rag.retrieval.retriever import retriever
        from rag.generation.llm_service import llm_service
        from rag.ingestion.pipeline import ingestion_pipeline

        self.reranker = reranker
        self.retriever = retriever
        self.llm_service = llm_service
        self.ingestion_pipeline = ingestion_pipeline

        logger.info("AppContext initialized successfully")


# Module-level singleton (eager loading)
context = AppContext()
