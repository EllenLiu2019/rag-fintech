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

logger = get_logger(__name__)


@lru_cache()
def get_llm_service() -> LLMService:
    """
    Get singleton LLM Service instance.

    Equivalent to:
        @Bean
        public LLMService llmService() {
            return new LLMService();
        }

    Returns:
        LLMService: Singleton instance for LLM operations
    """
    logger.info("Initializing LLMService singleton")
    return LLMService()


@lru_cache()
def get_retriever() -> Retriever:
    """
    Get singleton Retriever instance (default: dense mode).

    Equivalent to:
        @Bean
        public Retriever retriever() {
            return new Retriever();
        }

    Returns:
        Retriever: Singleton instance for vector search operations
    """
    logger.info("Initializing Retriever singleton (dense mode)")
    return Retriever()


@lru_cache()
def get_ingestion_pipeline() -> IngestionPipeline:
    """
    Get singleton IngestionPipeline instance.

    Equivalent to:
        @Bean
        public IngestionPipeline ingestionPipeline() {
            return new IngestionPipeline();
        }

    Returns:
        IngestionPipeline: Singleton instance for document ingestion operations
    """
    logger.info("Initializing IngestionPipeline singleton")
    return IngestionPipeline()
