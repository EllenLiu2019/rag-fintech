"""
RAG Server Entry Point
FastAPI application server startup script
"""

import uvicorn
from api import app  # noqa: E402, F401

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
