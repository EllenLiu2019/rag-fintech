"""
Dependency injection container for services.
Similar to Spring Boot's @Bean annotations.

All services are initialized as singletons using lru_cache.
"""

from functools import lru_cache
from typing import Dict
from rag.generation.llm_service import LLMService
from rag.retrieval.retriever import Retriever
from rag.ingestion.pipeline import IngestionPipeline
from common.log_utils import get_logger

logger = get_logger(__name__)

# Cache for retriever instances with different configurations
_retriever_cache: Dict[tuple, Retriever] = {}


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
    return Retriever(mode="dense")


def get_retriever_with_config(
    mode: str = "dense",
    enable_reranking: bool = False,
    fusion_method: str = "rrf",
) -> Retriever:
    """
    Get Retriever instance with specific configuration.
    Caches instances based on configuration to avoid re-initialization.

    Args:
        mode: Retrieval mode ("dense" or "hybrid")
        enable_reranking: Enable re-ranking (hybrid mode only)
        fusion_method: Fusion method ("rrf" or "weighted")

    Returns:
        Retriever: Configured retriever instance
    """
    # Create cache key from configuration
    cache_key = (mode, enable_reranking, fusion_method)

    # Return cached instance if exists
    if cache_key in _retriever_cache:
        return _retriever_cache[cache_key]

    # Create new instance
    logger.info(f"Initializing Retriever: mode={mode}, reranking={enable_reranking}, fusion={fusion_method}")
    retriever = Retriever(
        mode=mode,
        enable_reranking=enable_reranking,
        fusion_method=fusion_method,
    )

    # Cache the instance
    _retriever_cache[cache_key] = retriever

    return retriever


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
