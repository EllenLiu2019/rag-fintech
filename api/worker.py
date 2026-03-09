import os
import socket

from redis import Redis
from rq.worker import SimpleWorker, Worker

from common.config_utils import get_base_config
from common import get_logger, init_root_logger
from api.config import settings

init_root_logger(level=settings.LOG_LEVEL, format_str=settings.LOG_FORMAT)
logger = get_logger(__name__)


def main():
    redis_config = get_base_config("redis", {})
    redis_conn = Redis(
        host=redis_config.get("host"),
        port=int(redis_config.get("port")),
        username=redis_config.get("username"),
        password=redis_config.get("password"),
        decode_responses=False,
        socket_keepalive=True,
    )
    queue_name = redis_config.get("queue_name")

    worker_mode = os.getenv("RQ_WORKER_CLASS", "simple")
    if worker_mode == "fork":
        worker_class = Worker
    else:
        worker_class = SimpleWorker

    logger.info(f"Starting RQ worker: class={worker_class.__name__}, queue={queue_name}")
    worker = worker_class([queue_name], connection=redis_conn)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
