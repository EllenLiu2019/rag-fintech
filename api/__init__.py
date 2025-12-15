from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Initialize logging FIRST, before any other imports that might log
from api.config import settings
from common import get_logger, init_root_logger, setup_request_logging_middleware

init_root_logger(level=settings.LOG_LEVEL, format_str=settings.LOG_FORMAT)
logger = get_logger(__name__)

# Initialize app context (creates all services) before routers
from common.app_context import context  # noqa: E402, F401
from api.routers import document_api, chat_api, search_api  # noqa: E402
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

# Register API routes using APIRouter
app.include_router(document_api.router)
app.include_router(chat_api.router)
app.include_router(search_api.router)

# Register exception handlers (order matters: specific -> generic)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(RagBaseException, rag_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)
