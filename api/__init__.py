from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize logging FIRST, before any other imports that might log
from api.config import settings
from common import get_logger, init_root_logger, setup_request_logging_middleware

init_root_logger(level=settings.LOG_LEVEL, format_str=settings.LOG_FORMAT)
logger = get_logger(__name__)

# Initialize app context (creates all services) before routers
from api.routers import document_api, chat_api, search_api, claim_api  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from api.exception_handlers import (  # noqa: E402
    rag_exception_handler,
    generic_exception_handler,
    validation_exception_handler,
)
from common.exceptions import RagBaseException  # noqa: E402

logger.info("FastAPI application initialized")

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS middleware (using configuration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Add request logging middleware
setup_request_logging_middleware(app)

# Health check endpoint for K8S liveness/readiness probes
@app.get("/health", tags=["Health"])
async def health():
    from repository.rdb import rdb_client
    from repository.cache import redis_client

    checks = {"status": "ok", "service": settings.API_TITLE}

    try:
        rdb_client.engine.connect().close()
        checks["postgresql"] = "ok"
    except Exception:
        checks["postgresql"] = "error"
        checks["status"] = "degraded"

    try:
        if redis_client.redis_enabled:
            redis_client.client.ping()
            checks["redis"] = "ok"
        else:
            checks["redis"] = "disabled"
    except Exception:
        checks["redis"] = "error"
        checks["status"] = "degraded"

    status_code = 200 if checks["status"] == "ok" else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(content=checks, status_code=status_code)


# Register API routes using APIRouter
app.include_router(document_api.router)
app.include_router(chat_api.router)
app.include_router(search_api.router)
app.include_router(claim_api.router)

# Register exception handlers (order matters: specific -> generic)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(RagBaseException, rag_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)
