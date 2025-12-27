"""
Cache layer for performance optimization.
"""

from .redis_client import cached, redis_client

__all__ = ["cached", "redis_client"]
