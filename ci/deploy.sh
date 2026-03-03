#!/usr/bin/env bash
# Manual deploy script: build → push → apply to ACK
# Usage:
#    ./ci/deploy.sh                  # uses git HEAD as tag
#   IMAGE_TAG=v1.2.0  ./ci/deploy.sh # override tag
set -euo pipefail

# docker-compose-plugin v5 targets API 1.53 but Docker Engine 24.x supports max 1.43
export DOCKER_API_VERSION=1.43

# ── Configuration ──────
IMAGE_REGISTRY=${IMAGE_REGISTRY:-"crpi-jie2r6qhrtiyvvnq-vpc.cn-hangzhou.personal.cr.aliyuncs.com/ali-private"}
IMAGE_TAG=${IMAGE_TAG:-$(git rev-parse --short HEAD)}
FULL_IMAGE="${IMAGE_REGISTRY}/rag-fintech:${IMAGE_TAG}"
FULL_IMAGE_UI="${IMAGE_REGISTRY}/rag-fintech-ui:${IMAGE_TAG}"

# Public IP of this ECS, used as the API endpoint baked into the frontend build
ECS_PUBLIC_IP=${ECS_PUBLIC_IP:-"121.43.129.87"}
VITE_API_BASE_URL="http://${ECS_PUBLIC_IP}:8001"
# ────────────────────────

echo "==> Building backend image: ${FULL_IMAGE}"
docker build -f ci/Dockerfile -t "${FULL_IMAGE}" .

echo "==> Building frontend image: ${FULL_IMAGE_UI}"
docker build -f ci/Dockerfile.ui \
  --build-arg VITE_API_BASE_URL="${VITE_API_BASE_URL}" \
  -t "${FULL_IMAGE_UI}" .

echo "==> Logging in to ACR"
docker login \
  --username Ellen_lab \
  --password Happygirl123! \
  crpi-jie2r6qhrtiyvvnq-vpc.cn-hangzhou.personal.cr.aliyuncs.com

echo "==> Pushing images to ACR"
docker push "${FULL_IMAGE}"
docker push "${FULL_IMAGE_UI}"

# echo "==> Applying K8S manifests (IMAGE_TAG=${IMAGE_TAG})"
# export IMAGE_REGISTRY IMAGE_TAG
# for f in ci/k8s/*.yml; do
#   envsubst < "${f}" | kubectl apply -f -
# done

# echo "==> Waiting for rollout"
# kubectl rollout status deployment/rag-api    -n rag-fintech
# kubectl rollout status deployment/rag-worker -n rag-fintech

echo "==> Deploy complete: ${IMAGE_TAG}"
echo "    Backend:  ${FULL_IMAGE}"
echo "    Frontend: ${FULL_IMAGE_UI}"
