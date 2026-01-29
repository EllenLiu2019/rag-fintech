#!/bin/bash

GPU_ID=${1:-0}

MODEL_NAME="Qwen/Qwen3-32B-AWQ"

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Starting vLLM server on GPU $GPU_ID with model $MODEL_NAME..."
echo "Model cache location: $HF_HOME"

# --quantization awq: Use 4-bit weights to reduce VRAM usage and increase memory bandwidth efficiency
# --max-model-len: Limit context to reserve more memory for KV Cache (PagedAttention blocks)
# --gpu-memory-utilization: Reserve 90% VRAM for model weights + KV Cache
# --enforce-eager: Disable CUDA graph if you run into OOM during capture, but enabled (default) is faster
vllm serve $MODEL_NAME \
    --port 8000 \
    --max-model-len 12000 \
    --gpu-memory-utilization 0.9 \
    --quantization awq \
    --reasoning-parser qwen3 \
    --dtype auto \
    --enforce-eager \
    --kv-cache-dtype fp8 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes 
