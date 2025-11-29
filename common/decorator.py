import os
import threading
from functools import wraps

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