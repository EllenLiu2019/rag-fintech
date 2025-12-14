"""
Common decorators.
"""

import os
import threading
from functools import wraps

from common import get_logger

logger = get_logger(__name__)


def singleton(cls, *args, **kw):
    """
    Decorator to make a class a singleton.
    Thread-safe implementation with process ID support.
    """
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
