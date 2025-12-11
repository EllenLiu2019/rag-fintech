"""
Dependency injection container for services.
Similar to Spring Boot's @Bean annotations.

All services are initialized as singletons using lru_cache.
"""

from functools import lru_cache
from rag.generation.llm_service import LLMService
from rag.retrieval.retriever import Retriever
from rag.ingestion.pipeline import IngestionPipeline
from common.log_utils import get_logger
from common.model_registry import get_model_registry

logger = get_logger(__name__)


@lru_cache()
def get_llm_service() -> LLMService:
    """
    Get singleton LLM Service instance for QA tasks.
    Uses the "qa_reasoner" model for deep reasoning capabilities.
    """
    logger.info("Initializing LLMService singleton")
    registry = get_model_registry()
    model_config = registry.get_chat_model("qa_reasoner")
    return LLMService(model_config.to_dict())


@lru_cache()
def get_retriever() -> Retriever:
    """
    Get singleton Retriever instance.
    Uses the "query_lite" model for fast query optimization.
    """
    logger.info("Initializing Retriever singleton")
    registry = get_model_registry()
    model_config = registry.get_chat_model("query_lite")
    return Retriever(model_config.to_dict())


@lru_cache()
def get_ingestion_pipeline() -> IngestionPipeline:
    """
    Get singleton IngestionPipeline instance.
    Uses the "qa_lite" model for document processing.
    """
    logger.info("Initializing IngestionPipeline singleton")
    registry = get_model_registry()
    model_config = registry.get_chat_model("qa_lite")
    return IngestionPipeline(model_config.to_dict())
