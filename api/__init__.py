from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from common.log_middleware import setup_request_logging_middleware
from common.settings import init_settings
from api.document_api import upload_file, get_file_parsed, get_original_file, search_docs
from common.log_utils import get_logger, init_root_logger

init_root_logger(level=settings.LOG_LEVEL, format_str=settings.LOG_FORMAT)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Ensure initialization only happens once, even in reload mode.
    """
    try:
        init_settings()
        yield
    except Exception as e:
        logger.error(f"Error in FastAPI lifespan: {e}")
        raise
    finally:
        # TODO: shutdown cleanup if needed
        logger.info("FastAPI lifespan completed")


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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

# Register document API routes
app.post("/api/process")(upload_file)
app.get("/api/file-parsed")(get_file_parsed)
app.get("/api/file-original")(get_original_file)
app.post("/api/search")(search_docs)
