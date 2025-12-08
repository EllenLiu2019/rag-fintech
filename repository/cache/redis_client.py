import redis
import hashlib
import pickle
import inspect

from typing import Any, Optional, Callable
from functools import wraps

from common import settings
import logging

logger = logging.getLogger(__name__)


class RedisClient:

    def __init__(self):
        redis_config = settings.REDIS
        try:
            self.client = redis.Redis(
                host=redis_config.get("host"),
                port=redis_config.get("port"),
                decode_responses=redis_config.get("decode_responses", False),  # Must be False for pickle serialization
                username=redis_config.get("username"),
                password=redis_config.get("password"),
            )
            self.client.ping()

            logger.info(
                f"Redis client initialized successfully at {redis_config.get('host')}:{redis_config.get('port')}"
            )
        except (redis.ConnectionError, redis.TimeoutError) as e:
            logger.error(f"Redis connection failed: {e}.")
            raise e

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        # Create a string representation of arguments
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = "|".join(key_parts)

        logger.info(f"Key string: {key_str}")

        # Hash to create fixed-length key
        key_hash = hashlib.md5(key_str.encode()).hexdigest()

        return f"{prefix}:{key_hash}"

    def get(self, key: str) -> Optional[Any]:
        try:
            value = self.client.get(key)
            if value:
                return pickle.loads(value)
            return None
        except Exception as e:
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            serialized = pickle.dumps(value)
            if ttl:
                self.client.setex(key, ttl, serialized)
            else:
                self.client.set(key, serialized)
            return True
        except Exception as e:
            logger.warning(f"Cache set failed for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Cache delete failed for key {key}: {e}")
            return False

    def clear_prefix(self, prefix: str) -> int:
        """
        Clear all keys with a given prefix.

        Args:
            prefix: Key prefix to clear

        Returns:
            Number of keys deleted
        """
        try:
            pattern = f"{prefix}:*"
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Cache clear failed for prefix {prefix}: {e}")
            return 0

    def health(self) -> dict:
        """
        Check Redis health.

        Returns:
            Health status dict
        """
        try:
            self.client.ping()
            info = self.client.info()
            return {
                "type": "redis",
                "status": "green",
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "error": "",
            }
        except Exception as e:
            return {"type": "redis", "status": "red", "error": str(e)}


def cached(
    prefix: str,
    ttl: Optional[int] = None,
    key_func: Optional[Callable] = None,
):
    """
    Decorator for caching function results.

    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
        key_func: Optional function to generate cache key from arguments

    Example:
        @cached(prefix="embedding", ttl=3600)
        def embed_text(text: str):
            return expensive_embedding_call(text)
    """

    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        skip_first_arg = len(params) > 0 and params[0] in ("self", "cls")
        if skip_first_arg:
            logger.debug(
                f"Detected method '{func.__name__}' with '{params[0]}' parameter. "
                f"Will skip it when generating cache key."
            )

        @wraps(func)
        def wrapper(*args, **kwargs):
            redis_client = settings.REDIS_CLIENT

            cache_args = args[1:] if skip_first_arg else args

            # Generate cache key
            if key_func:
                cache_key = f"{prefix}:{key_func(*cache_args, **kwargs)}"
            else:
                cache_key = redis_client._generate_key(prefix, *cache_args, **kwargs)

            # Try to get from cache
            cached_value = redis_client.get(cache_key)
            if cached_value is not None:
                logger.info(f"Cache hit: {cache_key}")
                return cached_value

            # Cache miss - call function
            logger.info(f"Cache miss: {cache_key}")
            result = func(*args, **kwargs)

            # Store in cache
            redis_client.set(cache_key, result, ttl=ttl)

            return result

        return wrapper

    return decorator


def get_cache() -> Optional[RedisClient]:
    """
    Get Redis cache instance if enabled.

    Returns:
        RedisClient instance or None if cache is disabled
    """
    try:
        if hasattr(settings, "REDIS_CLIENT") and settings.REDIS_CLIENT:
            return settings.REDIS_CLIENT
        return None
    except Exception as e:
        logger.warning(f"Failed to get cache instance: {e}")
        return None
