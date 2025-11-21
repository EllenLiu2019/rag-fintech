#!/bin/bash

# FastAPI 服务器启动脚本
# 确保使用虚拟环境中的 Python

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# 检查虚拟环境是否存在
if [ ! -d ".venv" ]; then
    echo "❌ 错误: 未找到虚拟环境 .venv"
    echo "请先创建虚拟环境: python3.12 -m venv .venv"
    exit 1
fi

# 激活虚拟环境并启动服务器
echo "🚀 启动 FastAPI 服务器..."
echo "📍 项目根目录: $PROJECT_ROOT"
echo "📍 使用虚拟环境: $PROJECT_ROOT/.venv"
echo "🐍 Python 路径: $PROJECT_ROOT/.venv/bin/python"
echo ""

"$PROJECT_ROOT/.venv/bin/python" -m uvicorn api.main:app --reload --port 8001 --host 0.0.0.0

