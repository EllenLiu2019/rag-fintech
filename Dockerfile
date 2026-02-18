# ==============================================================================
# Multi-purpose Dockerfile for rag-fintech
# Builds a single image used by both FastAPI (api) and RQ Worker (worker)
# The entrypoint is determined by the command at runtime (K8S / docker-compose)
# ==============================================================================

# ---------- Stage 1: Dependencies ----------
FROM python:3.12-slim AS deps

# System dependencies required by psycopg (PostgreSQL), lxml, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (leverages Docker layer cache)
COPY pyproject.toml .
RUN pip install --no-cache-dir . \
    && pip install --no-cache-dir ruamel.yaml

# Pre-download models to avoid runtime download latency
ENV HF_HOME=/app/.cache/huggingface

# Sparse embedding model: BAAI/bge-m3 (~2.2GB)
RUN python -c "from FlagEmbedding import BGEM3FlagModel; BGEM3FlagModel('BAAI/bge-m3', device='cpu', use_fp16=False)"

# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim AS runtime

# Runtime system libraries (libpq for psycopg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from deps stage
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy pre-downloaded HuggingFace model cache
COPY --from=deps /app/.cache /app/.cache
ENV HF_HOME=/app/.cache/huggingface

# Copy application code
COPY api/ ./api/
COPY agent/ ./agent/
COPY common/ ./common/
COPY graphrag/ ./graphrag/
COPY rag/ ./rag/
COPY repository/ ./repository/
COPY conf/ ./conf/
COPY pyproject.toml .

# Install the project package in editable-like mode (so imports work)
RUN pip install --no-cache-dir -e .

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO

EXPOSE 8001

# No default CMD -- specified by K8S deployment or docker-compose
# API:    uvicorn api:app --host 0.0.0.0 --port 8001 --workers 2
# Worker: python api/worker.py
