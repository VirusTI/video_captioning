#!/usr/bin/env bash
# Run NVIDIA Triton Inference Server with a model exported from this repo.
#
# This script prepares a Triton model repository under artifacts/ and launches
# tritonserver inside the official docker image.
#
# Usage:
#   bash scripts/run_triton_server.sh baseline onnx artifacts/onnx/baseline.onnx
#   bash scripts/run_triton_server.sh advanced onnx artifacts/onnx/advanced.onnx
#   bash scripts/run_triton_server.sh advanced trt  artifacts/trt/advanced/advanced.plan
#
# Notes:
# - Config templates live under configs/triton/*.pbtxt
# - This repo's ONNX exports use fixed batch size = 1. We configure Triton with
#   max_batch_size: 0 and include the leading 1 in dims.

set -euo pipefail

MODEL_TYPE=${1:-}
BACKEND=${2:-}
MODEL_PATH=${3:-}

if [[ -z "${MODEL_TYPE}" || -z "${BACKEND}" || -z "${MODEL_PATH}" ]]; then
  echo "Usage: $0 <baseline|advanced> <onnx|trt> <path_to_model_file>"
  exit 1
fi

MODEL_TYPE=$(echo "${MODEL_TYPE}" | tr '[:upper:]' '[:lower:]')
BACKEND=$(echo "${BACKEND}" | tr '[:upper:]' '[:lower:]')

if [[ "${MODEL_TYPE}" != "baseline" && "${MODEL_TYPE}" != "advanced" ]]; then
  echo "ERROR: model_type must be 'baseline' or 'advanced'"
  exit 1
fi

if [[ "${BACKEND}" != "onnx" && "${BACKEND}" != "trt" ]]; then
  echo "ERROR: backend must be 'onnx' or 'trt'"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: model file not found: ${MODEL_PATH}"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed (or not on PATH)."
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "ERROR: cannot talk to the Docker daemon (permission denied or daemon not running)."
  echo "Fix: add user to docker group (sudo usermod -aG docker $USER) and re-login, or run via sudo."
  exit 1
fi

TEMPLATE="configs/triton/${MODEL_TYPE}_${BACKEND}.pbtxt"
if [[ ! -f "${TEMPLATE}" ]]; then
  echo "ERROR: Triton config template not found: ${TEMPLATE}"
  exit 1
fi

REPO_DIR=${REPO_DIR:-"artifacts/triton_model_repo"}
MODEL_DIR="${REPO_DIR}/${MODEL_TYPE}"
VERSION_DIR="${MODEL_DIR}/1"
mkdir -p "${VERSION_DIR}"

MODEL_ABS=$(realpath "${MODEL_PATH}")
REPO_ABS=$(realpath "${REPO_DIR}")

if [[ "${BACKEND}" == "onnx" ]]; then
  cp -f "${MODEL_ABS}" "${VERSION_DIR}/model.onnx"
else
  cp -f "${MODEL_ABS}" "${VERSION_DIR}/model.plan"
fi

cp -f "${TEMPLATE}" "${MODEL_DIR}/config.pbtxt"

echo "[INFO] Triton model repo: ${REPO_ABS}"

echo "[INFO] Starting Triton on ports 8000(http), 8001(grpc), 8002(metrics)"

echo "[INFO] Image: ${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.09-py3}"

docker run --rm --gpus=all \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v "${REPO_ABS}:/models" \
  "${TRITON_IMAGE:-nvcr.io/nvidia/tritonserver:24.09-py3}" \
  tritonserver --model-repository=/models
