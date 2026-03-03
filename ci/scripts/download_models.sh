#!/usr/bin/env bash
# One-time script to pre-download model weights to the host.
# Run this BEFORE starting docker compose for the first time.
#
# Usage:
#   bash ci/scripts/download_models.sh
#
# Models are saved to /data/hf_cache and bind-mounted into containers.
# Re-running is safe: already-downloaded files are skipped.
set -euo pipefail

HF_CACHE_DIR=${HF_CACHE_DIR:-/data/hf_cache}
HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
IMAGE=${IMAGE:-crpi-jie2r6qhrtiyvvnq-vpc.cn-hangzhou.personal.cr.aliyuncs.com/ali-private/rag-fintech:latest}

mkdir -p "$HF_CACHE_DIR"
echo "==> Downloading models to $HF_CACHE_DIR (endpoint: $HF_ENDPOINT)"

DOCKER_API_VERSION=1.43 docker run --rm \
  -e HF_ENDPOINT="$HF_ENDPOINT" \
  -e HF_HOME=/cache \
  -v "$HF_CACHE_DIR":/cache \
  "$IMAGE" \
  python - <<'PYEOF'
from huggingface_hub import snapshot_download

models = [
    "BAAI/bge-m3",
]

for model in models:
    print(f"Downloading {model} ...")
    snapshot_download(
        model,
        ignore_patterns=["*.DS_Store", "imgs/*", "*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print(f"  {model} done.")

print("All models downloaded.")
PYEOF

echo "==> Done. Models are in $HF_CACHE_DIR"
