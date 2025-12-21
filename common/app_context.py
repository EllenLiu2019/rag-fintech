"""
Application Context - Centralized service container.
All services are explicitly initialized at module load time (eager loading).
"""

from common.log_utils import get_logger

logger = get_logger(__name__)


class AppContext:
    """
    Application-wide service container.
    """

    def __init__(self):

        # RAG services (these are module-level singletons)
        from rag.retrieval import retriever  # noqa: F401
        from rag.generation import llm_service  # noqa: F401
        from rag.ingestion import ingestion_pipeline  # noqa: F401

        logger.info("AppContext initialized successfully")


# Module-level singleton (eager loading)
context = AppContext()
