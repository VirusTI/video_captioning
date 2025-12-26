#!/usr/bin/env bash
# Build a TensorRT engine (.plan) from an exported ONNX model.
#
# Usage:
#   bash scripts/build_trt_engine.sh artifacts/onnx/baseline.onnx
#   bash scripts/build_trt_engine.sh artifacts/onnx/advanced.onnx
#
# Notes:
# - This uses NVIDIA's TensorRT docker image and `trtexec`.
# - Baseline ONNX has a dynamic sequence length dimension for `tokens`.
#   We provide a min/opt/max profile via environment variables.
# - Advanced ONNX export uses fixed `NUM_FRAMES` and `SEQ_LEN` at export time.
#   Those must match here.

set -euo pipefail

ONNX_PATH=${1:-}
if [[ -z "${ONNX_PATH}" ]]; then
  echo "Usage: $0 <path_to_onnx> [--fp16|--fp32]"
  exit 1
fi

shift || true

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "ERROR: ONNX file not found: ${ONNX_PATH}"
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: docker is not installed (or not on PATH)."
  exit 1
fi

# Docker may be installed but not usable by the current user (common on Linux).
# Fail early with actionable instructions.
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: cannot talk to the Docker daemon (permission denied or daemon not running)."
  echo ""
  echo "Fix options (choose one):"
  echo "  1) Add your user to the docker group and re-login:" \
       "sudo usermod -aG docker $USER"
  echo "  2) Run this script with sudo (if allowed):" \
       "sudo bash scripts/build_trt_engine.sh <path_to_onnx>"
  echo "  3) Ensure the docker service is running: sudo systemctl start docker"
  exit 1
fi

ONNX_ABS=$(realpath "${ONNX_PATH}")
ONNX_DIR=$(dirname "${ONNX_ABS}")
ONNX_FILE=$(basename "${ONNX_ABS}")
MODEL_NAME=$(basename "${ONNX_FILE}" .onnx)

# Output location (kept under artifacts/, which is ignored by git).
OUTPUT_DIR=${OUTPUT_DIR:-"artifacts/trt/${MODEL_NAME}"}
mkdir -p "${OUTPUT_DIR}"
OUTPUT_ABS=$(realpath "${OUTPUT_DIR}")

# Model input tensor names are defined in video_captioning/onnx/export.py:
# - baseline:  frames, tokens
# - advanced:  frames, tokens, frame_mask

IMAGE_SIZE=${IMAGE_SIZE:-224}
BATCH_SIZE=${BATCH_SIZE:-1}

# If the script is executed via sudo, keep output file ownership under the original user.
HOST_UID=${SUDO_UID:-$(id -u)}
HOST_GID=${SUDO_GID:-$(id -g)}

# If we are root (typically via sudo), ensure the mounted output directory is
# writable by the user we will run the container as.
if [[ "$(id -u)" == "0" ]]; then
  chown -R "${HOST_UID}:${HOST_GID}" "${OUTPUT_ABS}" || true
fi

# Advanced export uses fixed sizes; keep defaults aligned with convert_onnx.
NUM_FRAMES=${NUM_FRAMES:-16}
SEQ_LEN=${SEQ_LEN:-16}

# Baseline export keeps tokens seq_len dynamic; provide TRT optimization profile.
TOKENS_MIN_LEN=${TOKENS_MIN_LEN:-16}
TOKENS_OPT_LEN=${TOKENS_OPT_LEN:-16}
TOKENS_MAX_LEN=${TOKENS_MAX_LEN:-64}

# Precision control.
# Default: fp16 enabled.
FP16_DEFAULT=${FP16:-1}
FP16=${FP16_DEFAULT}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fp16)
      FP16=1
      ;;
    --fp32|--no-fp16)
      FP16=0
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      echo "Usage: $0 <path_to_onnx> [--fp16|--fp32]"
      exit 1
      ;;
  esac
  shift
done

TRT_IMAGE=${TRT_IMAGE:-"nvcr.io/nvidia/tensorrt:24.09-py3"}

TRT_FLAGS=(
  "--onnx=/onnx/${ONNX_FILE}"
  "--saveEngine=/output/${MODEL_NAME}.plan"
)

if [[ "${FP16}" == "1" ]]; then
  TRT_FLAGS+=("--fp16")
fi

if [[ "${MODEL_NAME}" == *"advanced"* ]]; then
  # Advanced ONNX export uses fixed shapes (dynamic_axes = {}).
  # For a fully static network, `trtexec` rejects explicit `--shapes`.
  # The input tensor shapes will be taken directly from the ONNX graph.
  :
else
  # Baseline: dynamic seq_len for tokens.
  TRT_FLAGS+=(
    "--minShapes=frames:${BATCH_SIZE}x3x${IMAGE_SIZE}x${IMAGE_SIZE},tokens:${BATCH_SIZE}x${TOKENS_MIN_LEN}"
    "--optShapes=frames:${BATCH_SIZE}x3x${IMAGE_SIZE}x${IMAGE_SIZE},tokens:${BATCH_SIZE}x${TOKENS_OPT_LEN}"
    "--maxShapes=frames:${BATCH_SIZE}x3x${IMAGE_SIZE}x${IMAGE_SIZE},tokens:${BATCH_SIZE}x${TOKENS_MAX_LEN}"
  )
fi

echo "[INFO] Converting: ${ONNX_ABS}"
echo "[INFO] Output: ${OUTPUT_ABS}/${MODEL_NAME}.plan"
echo "[INFO] TensorRT image: ${TRT_IMAGE}"

docker run --rm --gpus=all \
  --user "${HOST_UID}:${HOST_GID}" \
  -v "${ONNX_DIR}:/onnx" \
  -v "${OUTPUT_ABS}:/output" \
  "${TRT_IMAGE}" \
  trtexec "${TRT_FLAGS[@]}"

echo "[INFO] Done: ${OUTPUT_ABS}/${MODEL_NAME}.plan"
