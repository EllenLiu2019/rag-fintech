import hashlib
import pickle
import socket
from typing import Any, Optional, Callable
from functools import wraps
import inspect
import time
import asyncio

from redis import Redis, RedisError, TimeoutError
from rq import Queue, Retry
from rq.job import Job
from rq.exceptions import NoSuchJobError

from common.decorator import singleton
from common.config_utils import get_base_config
from common import get_logger

logger = get_logger(__name__)


@singleton
class RedisClient:

    def __init__(self, config: dict = None):
        # Support both DI and standalone usage
        redis_config = config or get_base_config("redis", {})
        self.redis_enabled = redis_config.get("enable", True)
        if not self.redis_enabled:
            logger.info("Redis is disabled. Skipping Redis client initialization.")
            return

        try:
            self.client = Redis(
                host=redis_config.get("host"),
                port=int(redis_config.get("port")),
                username=redis_config.get("username"),
                password=redis_config.get("password"),
                decode_responses=redis_config.get("decode_responses", False),  # Must be False for pickle serialization
                socket_keepalive=True,
                socket_keepalive_options={
                    socket.TCP_KEEPIDLE: 60,    # 空闲 60s 后开始发探测（早于 NAT 超时）
                    socket.TCP_KEEPINTVL: 10,   # 每 10s 发一次探测
                    socket.TCP_KEEPCNT: 3,      # 连续 3 次无响应则断开
                },
            )
            self.client.ping()
            queue_name = redis_config.get("queue_name", "default")
            self.queue = Queue(name=queue_name, connection=self.client)

            logger.info(
                f"Redis client initialized successfully at {redis_config.get('host')}:{redis_config.get('port')}, "
                f"queue: {queue_name}"
            )
        except (RedisError, TimeoutError) as e:
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

    def enqueue(
        self,
        func: Callable,
        *args,
        job_timeout: int = 600,
        result_ttl: int = 3600,
        failure_ttl: int = 86400,
        **kwargs,
    ) -> Job:
        """
        Enqueue a job to RQ queue.

        Returns:
            RQ Job object
        """
        return self.queue.enqueue(
            func,
            *args,
            retry=Retry(max=1, interval=60),
            job_timeout=job_timeout,
            result_ttl=result_ttl,
            failure_ttl=failure_ttl,
            **kwargs,
        )

    def get_job(self, job_id: str) -> Optional[Job]:
        try:
            return Job.fetch(job_id, connection=self.client)
        except NoSuchJobError:
            logger.warning(f"Job {job_id} not found")
            return None

    def update_progress(self, job_id: str, step: int, message: str):
        """Update job progress in Redis."""
        try:
            key = f"job:{job_id}:progress"
            result = self.client.hset(
                key,
                mapping={
                    "step": step,
                    "message": message,
                    "updated_at": time.time(),
                },
            )
            logger.debug(f"Updated progress in Redis: key={key}, step={step}, message={message}, result={result}")
            return result
        except Exception as e:
            logger.error(f"Failed to update job progress {job_id}: {e}", exc_info=True)

    def get_progress(self, job_id: str):
        try:
            key = f"job:{job_id}:progress"
            result = self.client.hgetall(key)
            logger.debug(f"Get progress from Redis: key={key}, result={result}")
            return result
        except Exception as e:
            logger.error(f"Failed to get progress for job_id {job_id}: {e}", exc_info=True)
            return {}


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
            if not redis_client.redis_enabled:
                logger.info("Redis is disabled. Skipping cache.")
                return func(*args, **kwargs)

            cache_args = args[1:] if skip_first_arg else args

            # Generate cache key
            if key_func:
                # Pass full args to key_func (including self for methods)
                cache_key = f"{prefix}:{key_func(*args, **kwargs)}"
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


def async_cached(prefix: str, ttl: int = None, key_func: Callable = None):
    def decorator(func):
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        skip_first_arg = len(params) > 0 and params[0] in ("self", "cls")
        if skip_first_arg:
            logger.debug(
                f"Detected method '{func.__name__}' with '{params[0]}' parameter. "
                f"Will skip it when generating cache key."
            )

        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not redis_client.redis_enabled:
                logger.info("Redis is disabled. Skipping cache.")
                return func(*args, **kwargs)

            cache_args = args[1:] if skip_first_arg else args
            if key_func:
                cache_key = f"{prefix}:{key_func(*args, **kwargs)}"
            else:
                cache_key = redis_client._generate_key(prefix, *cache_args, **kwargs)

            # async Redis get
            cached_value = await asyncio.to_thread(redis_client.get, cache_key)
            if cached_value is not None:
                return cached_value
            result = await func(*args, **kwargs)
            await asyncio.to_thread(redis_client.set, cache_key, result, ttl=ttl)
            return result

        return wrapper

    return decorator


def _create_redis_client() -> RedisClient:
    return RedisClient()


redis_client = _create_redis_client()
