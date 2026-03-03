# ==============================================================================
# Multi-purpose Dockerfile for rag-fintech
# Builds a single image used by both FastAPI (api) and RQ Worker (worker)
# The entrypoint is determined by the command at runtime (K8S / docker-compose)
# ==============================================================================

# ---------- Stage 1: Dependencies ----------
FROM python:3.12-slim AS deps

WORKDIR /app

# Install Python dependencies first (leverages Docker layer cache)
#  stub __init__.py
# files satisfy setuptools package discovery without copying real source yet.
COPY pyproject.toml ./
RUN mkdir -p api agent benchmark common graphrag rag repository && \
    touch api/__init__.py agent/__init__.py benchmark/__init__.py \
          common/__init__.py graphrag/__init__.py rag/__init__.py repository/__init__.py

# Install CPU-only torch from pre-downloaded wheel (avoids slow download.pytorch.org CDN).
# Run `make wheels` or `pip download torch --no-deps --index-url
# https://download.pytorch.org/whl/cpu -d wheels/` on the host before building.
COPY wheels/ ./wheels/
RUN pip install --no-cache-dir ./wheels/torch*.whl && \
    pip install --no-cache-dir -i http://mirrors.cloud.aliyuncs.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com . && \
    rm -rf ./wheels/

# Model weights are NOT baked into the image.
# Mount a host volume to HF_HOME at runtime to cache downloaded models:
#   docker run -v /data/hf_cache:/app/.cache/huggingface ...
ENV HF_HOME=/app/.cache/huggingface
ENV HF_ENDPOINT=https://hf-mirror.com

# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim AS runtime

# Use Aliyun mirror for faster apt downloads in China
RUN sed -i 's|http://deb.debian.org|http://mirrors.cloud.aliyuncs.com|g' /etc/apt/sources.list.d/debian.sources

# Runtime system libraries (libpq for psycopg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from deps stage
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

ENV HF_HOME=/app/.cache/huggingface
ENV HF_ENDPOINT=https://hf-mirror.com

# Copy application code
COPY api/ ./api/
COPY agent/ ./agent/
COPY benchmark/ ./benchmark/
COPY common/ ./common/
COPY graphrag/ ./graphrag/
COPY rag/ ./rag/
COPY repository/ ./repository/
COPY conf/ ./conf/
COPY pyproject.toml ./

# Source is in /app; PYTHONPATH lets Python find all packages and subpackages
# directly from the source tree without reinstalling into site-packages
# (which would shadow /app/api with an incomplete installation).
ENV PYTHONPATH=/app

# Environment defaults
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO

EXPOSE 8001

# No default CMD -- specified by K8S deployment
# API:    uvicorn api:app --host 0.0.0.0 --port 8001 --workers 2
# Worker: python api/worker.py
