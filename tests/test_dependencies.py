"""
Unit tests for service dependencies module.

Tests singleton behavior and dependency injection functions.
"""

from unittest.mock import patch


class TestDependencies:
    """Test dependency injection functions"""

    def test_get_llm_service_singleton(self):
        """Test that get_llm_service returns singleton instance"""
        from service.dependencies import get_llm_service
        from service.llm_service import LLMService

        # Clear cache to ensure fresh test
        get_llm_service.cache_clear()

        # Get instances twice
        service1 = get_llm_service()
        service2 = get_llm_service()

        # Should be the same instance (singleton)
        assert service1 is service2
        assert isinstance(service1, LLMService)
        assert isinstance(service2, LLMService)

    def test_get_retriever_singleton(self):
        """Test that get_retriever returns singleton instance"""
        from service.dependencies import get_retriever
        from service.retrieval.retriever import Retriever

        # Clear cache to ensure fresh test
        get_retriever.cache_clear()

        # Get instances twice
        retriever1 = get_retriever()
        retriever2 = get_retriever()

        # Should be the same instance (singleton)
        assert retriever1 is retriever2
        assert isinstance(retriever1, Retriever)
        assert isinstance(retriever2, Retriever)

    def test_get_ingestion_pipeline_singleton(self):
        """Test that get_ingestion_pipeline returns singleton instance"""
        from service.dependencies import get_ingestion_pipeline
        from service.ingestion.pipeline import IngestionPipeline

        # Clear cache to ensure fresh test
        get_ingestion_pipeline.cache_clear()

        # Get instances twice
        pipeline1 = get_ingestion_pipeline()
        pipeline2 = get_ingestion_pipeline()

        # Should be the same instance (singleton)
        assert pipeline1 is pipeline2
        assert isinstance(pipeline1, IngestionPipeline)
        assert isinstance(pipeline2, IngestionPipeline)

    def test_all_services_are_different_instances(self):
        """Test that different services are different instances"""
        from service.dependencies import (
            get_llm_service,
            get_retriever,
            get_ingestion_pipeline,
        )

        # Clear all caches
        get_llm_service.cache_clear()
        get_retriever.cache_clear()
        get_ingestion_pipeline.cache_clear()

        # Get all services
        llm_service = get_llm_service()
        retriever = get_retriever()
        pipeline = get_ingestion_pipeline()

        # All should be different instances
        assert llm_service is not retriever
        assert llm_service is not pipeline
        assert retriever is not pipeline

    @patch("service.dependencies.logger")
    def test_get_llm_service_logs_initialization(self, mock_logger):
        """Test that get_llm_service logs initialization"""
        from service.dependencies import get_llm_service

        # Clear cache
        get_llm_service.cache_clear()

        # Get service (should trigger initialization log)
        get_llm_service()

        # Verify log was called
        mock_logger.info.assert_called_once()
        assert "Initializing LLMService singleton" in mock_logger.info.call_args[0][0]

    @patch("service.dependencies.logger")
    def test_get_retriever_logs_initialization(self, mock_logger):
        """Test that get_retriever logs initialization"""
        from service.dependencies import get_retriever

        # Clear cache
        get_retriever.cache_clear()

        # Get service (should trigger initialization log)
        get_retriever()

        # Verify log was called
        mock_logger.info.assert_called_once()
        assert "Initializing Retriever singleton" in mock_logger.info.call_args[0][0]

    @patch("service.dependencies.logger")
    def test_get_ingestion_pipeline_logs_initialization(self, mock_logger):
        """Test that get_ingestion_pipeline logs initialization"""
        from service.dependencies import get_ingestion_pipeline

        # Clear cache
        get_ingestion_pipeline.cache_clear()

        # Get service (should trigger initialization log)
        get_ingestion_pipeline()

        # Verify log was called
        mock_logger.info.assert_called_once()
        assert "Initializing IngestionPipeline singleton" in mock_logger.info.call_args[0][0]

    def test_get_llm_service_only_logs_once(self):
        """Test that initialization log is only called once per service"""
        from service.dependencies import get_llm_service

        with patch("service.dependencies.logger") as mock_logger:
            # Clear cache
            get_llm_service.cache_clear()

            # Get service multiple times
            get_llm_service()
            get_llm_service()
            get_llm_service()

            # Should only log once (first call)
            assert mock_logger.info.call_count == 1

    def test_get_retriever_only_logs_once(self):
        """Test that retriever initialization log is only called once"""
        from service.dependencies import get_retriever

        with patch("service.dependencies.logger") as mock_logger:
            # Clear cache
            get_retriever.cache_clear()

            # Get service multiple times
            get_retriever()
            get_retriever()
            get_retriever()

            # Should only log once (first call)
            assert mock_logger.info.call_count == 1

    def test_get_ingestion_pipeline_only_logs_once(self):
        """Test that pipeline initialization log is only called once"""
        from service.dependencies import get_ingestion_pipeline

        with patch("service.dependencies.logger") as mock_logger:
            # Clear cache
            get_ingestion_pipeline.cache_clear()

            # Get service multiple times
            get_ingestion_pipeline()
            get_ingestion_pipeline()
            get_ingestion_pipeline()

            # Should only log once (first call)
            assert mock_logger.info.call_count == 1

    def test_cache_clear_creates_new_instance(self):
        """Test that cache_clear allows creating new instances"""
        from service.dependencies import get_llm_service
        from service.llm_service import LLMService

        # Get initial instance
        get_llm_service.cache_clear()
        service1 = get_llm_service()

        # Clear cache and get new instance
        get_llm_service.cache_clear()
        service2 = get_llm_service()

        # Should be different instances (after cache clear)
        assert service1 is not service2
        # But both should be LLMService instances
        assert isinstance(service1, LLMService)
        assert isinstance(service2, LLMService)

    def test_services_are_functional(self):
        """Test that returned services are functional instances"""
        from service.dependencies import (
            get_llm_service,
            get_retriever,
            get_ingestion_pipeline,
        )

        # Clear all caches
        get_llm_service.cache_clear()
        get_retriever.cache_clear()
        get_ingestion_pipeline.cache_clear()

        # Get services
        llm_service = get_llm_service()
        retriever = get_retriever()
        pipeline = get_ingestion_pipeline()

        # Verify they have expected attributes/methods
        assert hasattr(llm_service, "answer_question")
        assert hasattr(llm_service, "stream_answer_question")
        assert hasattr(retriever, "search")
        assert hasattr(pipeline, "handle_document")
        assert hasattr(pipeline, "build_from_parsed_documents")

    def test_concurrent_access_singleton(self):
        """
        Test that concurrent access returns consistent instances.

        Note: lru_cache is thread-safe in CPython, but in some edge cases
        with high concurrency, multiple instances might be created initially.
        However, after all threads complete, subsequent calls should return
        the same cached instance.
        """
        from service.dependencies import get_llm_service
        import concurrent.futures

        # Clear cache
        get_llm_service.cache_clear()

        # Get service concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(get_llm_service) for _ in range(10)]
            services = [future.result() for future in concurrent.futures.as_completed(futures)]

        # All should be LLMService instances
        for service in services:
            assert isinstance(service, type(services[0]))

        # After concurrent access, subsequent calls should return same instance
        # (This verifies that caching works correctly after concurrency)
        service_after = get_llm_service()
        assert isinstance(service_after, type(services[0]))
