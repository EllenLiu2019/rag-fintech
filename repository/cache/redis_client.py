import redis
import hashlib
import pickle
import inspect

from typing import Any, Optional, Callable
from functools import wraps

from common import config
import logging

logger = logging.getLogger(__name__)


class RedisClient:

    def __init__(self):
        redis_config = config.REDIS
        try:
            self.client = redis.Redis(
                host=redis_config.get("host"),
                port=int(redis_config.get("port")),
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

        logger.info(f"Key string: {key_str[:100]}")

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
