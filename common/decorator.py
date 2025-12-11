import os
import threading
from functools import wraps
from typing import Optional, Callable
import inspect
from common import config
import logging

logger = logging.getLogger(__name__)


def singleton(cls, *args, **kw):
    instances = {}
    lock = threading.Lock()

    @wraps(cls)
    def _singleton():
        key = str(cls) + str(os.getpid())

        if key in instances:
            return instances[key]

        with lock:
            if key not in instances:
                instances[key] = cls(*args, **kw)
            return instances[key]

    return _singleton


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
            redis_client = config.REDIS_CLIENT

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
