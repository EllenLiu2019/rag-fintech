#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

echo "🚀 Starting FastAPI server..."
echo "📍 Project root: $PROJECT_ROOT"
echo "🐍 Python path: $PROJECT_ROOT/.venv/bin/python"
echo ""

"$PROJECT_ROOT/.venv/bin/python" -m uvicorn api.main:app --reload --port 8001 --host 0.0.0.0

