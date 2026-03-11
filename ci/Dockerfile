# ==============================================================================
# Multi-purpose Dockerfile for rag-fintech
# Builds a single image used by both FastAPI (api) and RQ Worker (worker)
# The entrypoint is determined by the command at runtime (K8S / docker-compose)
# ==============================================================================

# ---------- Stage 1: Dependencies ----------
FROM python:3.12-slim AS deps

WORKDIR /app

COPY pyproject.toml ./
RUN mkdir -p api agent benchmark common graphrag rag repository && \
    touch api/__init__.py agent/__init__.py benchmark/__init__.py \
          common/__init__.py graphrag/__init__.py rag/__init__.py repository/__init__.py
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir .

# ---------- Stage 2: Runtime ----------
FROM python:3.12-slim AS runtime

# for psycopg, install libpq5 and remove apt lists(index files)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

COPY . .

ENV PYTHONPATH=/app