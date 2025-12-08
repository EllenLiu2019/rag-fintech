"""
Test cached decorator with class methods.

Tests the improvement that automatically skips 'self' parameter when generating cache keys.
"""

import pytest
from unittest.mock import patch


class MockRedisClient:
    """Mock Redis client for testing"""

    def __init__(self):
        self.cache = {}
        self.get_count = 0
        self.set_count = 0

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        import hashlib

        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = "|".join(key_parts)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"{prefix}:{key_hash}"

    def get(self, key: str):
        self.get_count += 1
        return self.cache.get(key)

    def set(self, key: str, value, ttl=None):
        self.set_count += 1
        self.cache[key] = value
        return True


# Import the cached decorator
from repository.cache.redis_client import cached


class Calculator:
    """Test class with cached methods"""

    def __init__(self, name: str):
        self.name = name
        self.call_count = 0

    @cached(prefix="calc", ttl=60)
    def add(self, a: int, b: int) -> int:
        """Cached addition method"""
        self.call_count += 1
        return a + b

    @cached(prefix="multiply", ttl=60)
    def multiply(self, a: int, b: int, factor: int = 1) -> int:
        """Cached multiplication with optional parameter"""
        self.call_count += 1
        return a * b * factor


class Retriever:
    """Mock retriever class similar to actual implementation"""

    def __init__(self):
        self.search_count = 0

    @cached(prefix="search", ttl=1800)
    def search(self, query: str, kb_id: str = "default", top_k: int = 5):
        """Simulates retriever.search method"""
        self.search_count += 1
        return {"query": query, "kb_id": kb_id, "results": [f"result_{i}" for i in range(top_k)]}


@pytest.fixture
def mock_redis():
    """Fixture to provide mock Redis client"""
    mock_client = MockRedisClient()

    # Mock settings.REDIS_CLIENT
    with patch("common.settings.REDIS_CLIENT", mock_client):
        yield mock_client


def test_cache_same_instance(mock_redis):
    """
    Test: Same instance, same parameters should hit cache
    """
    calc = Calculator("calc1")

    # First call - should execute
    result1 = calc.add(1, 2)
    assert result1 == 3
    assert calc.call_count == 1
    assert mock_redis.set_count == 1

    # Second call - should hit cache
    result2 = calc.add(1, 2)
    assert result2 == 3
    assert calc.call_count == 1  # ← Not increased, cache hit
    assert mock_redis.get_count == 2  # Both calls checked cache


def test_cache_different_instances(mock_redis):
    """
    Test: Different instances, same parameters should share cache

    This is the KEY improvement - before fix, this would fail!
    """
    calc1 = Calculator("calc1")
    calc2 = Calculator("calc2")

    # First call with calc1
    result1 = calc1.add(1, 2)
    assert result1 == 3
    assert calc1.call_count == 1

    # Second call with calc2 (different instance, same params)
    result2 = calc2.add(1, 2)
    assert result2 == 3
    assert calc2.call_count == 0  # ✅ Should hit calc1's cache


def test_cache_different_params(mock_redis):
    """
    Test: Same instance, different parameters should NOT hit cache
    """
    calc = Calculator("calc1")

    result1 = calc.add(1, 2)
    assert calc.call_count == 1

    result2 = calc.add(3, 4)  # Different params
    assert calc.call_count == 2  # Should execute again

    result3 = calc.add(1, 2)  # Same as first call
    assert calc.call_count == 2  # Should hit cache


def test_cache_with_kwargs(mock_redis):
    """
    Test: Methods with keyword arguments
    """
    calc1 = Calculator("calc1")
    calc2 = Calculator("calc2")

    # Call with keyword argument
    result1 = calc1.multiply(2, 3, factor=5)
    assert result1 == 30
    assert calc1.call_count == 1

    # Different instance, same params (including kwargs)
    result2 = calc2.multiply(2, 3, factor=5)
    assert result2 == 30
    assert calc2.call_count == 0  # ✅ Should hit cache

    # Different keyword argument value
    result3 = calc2.multiply(2, 3, factor=10)
    assert result3 == 60
    assert calc2.call_count == 1  # Should execute


def test_retriever_cache_across_instances(mock_redis):
    """
    Test: Retriever instances should share cache

    This simulates the actual use case in the project.
    """
    retriever1 = Retriever()
    retriever2 = Retriever()

    # First search with retriever1
    results1 = retriever1.search("保障范围", kb_id="default_kb", top_k=5)
    assert retriever1.search_count == 1
    assert len(results1["results"]) == 5

    # Same search with retriever2 (different instance)
    results2 = retriever2.search("保障范围", kb_id="default_kb", top_k=5)
    assert retriever2.search_count == 0  # ✅ Should hit cache
    assert results2 == results1

    # Different query
    results3 = retriever2.search("理赔流程", kb_id="default_kb", top_k=5)
    assert retriever2.search_count == 1  # Should execute


def test_cache_key_generation(mock_redis):
    """
    Test: Verify cache keys are generated without 'self'
    """
    calc1 = Calculator("calc1")
    calc2 = Calculator("calc2")

    # Capture cache keys
    original_set = mock_redis.set
    captured_keys = []

    def capturing_set(key, value, ttl=None):
        captured_keys.append(key)
        return original_set(key, value, ttl)

    mock_redis.set = capturing_set

    # Make calls
    calc1.add(1, 2)
    calc2.add(1, 2)

    # Both should generate the same cache key
    assert len(captured_keys) == 1  # Only one set (second call hit cache)
    print(f"Cache key: {captured_keys[0]}")

    # Verify key doesn't contain instance info
    assert "Calculator" not in captured_keys[0]  # No class name in key


def test_cache_performance_improvement(mock_redis):
    """
    Test: Measure cache hit rate improvement
    """
    # Simulate multiple requests with different instances (like in API)
    instances = [Retriever() for _ in range(10)]

    # Execute same query across all instances
    query = "保障范围"
    for retriever in instances:
        retriever.search(query, kb_id="default_kb", top_k=5)

    # With improved caching:
    # - Only first call should execute
    # - Rest should hit cache
    total_executions = sum(r.search_count for r in instances)
    cache_hit_rate = (10 - total_executions) / 10

    assert total_executions == 1  # Only first call should execute
    assert cache_hit_rate == 0.9  # 90% cache hit rate
