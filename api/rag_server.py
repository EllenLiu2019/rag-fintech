"""
RAG Server Entry Point
FastAPI application server startup script
"""

import os
import sys
import uvicorn

# Ensure project root is in Python path
# This is needed when running the script directly (not via module)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from api import app  # noqa: E402, F401

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )
