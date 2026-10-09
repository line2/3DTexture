#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET="${1:-${ROOT_DIR}/dataset/rgbd_dataset_freiburg1_teddy}"
BUILD_DIR="${2:-${ROOT_DIR}/build/Release}"
EXECUTABLE="${BUILD_DIR}/3DTexture"

if [[ ! -x "${EXECUTABLE}" ]]; then
    "${ROOT_DIR}/scripts/build.sh"
fi
if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Executable not found: ${EXECUTABLE}" >&2
    exit 1
fi

OBJ_FILE="${DATASET}/mesh.obj"
VIEW_DIR="${DATASET}/rgb_cam"
OUT_DIR="${DATASET}/out"

if [[ ! -f "${OBJ_FILE}" ]]; then
    echo "Mesh file not found: ${OBJ_FILE}" >&2
    exit 1
fi
if [[ ! -d "${VIEW_DIR}" ]]; then
    echo "View directory not found: ${VIEW_DIR}" >&2
    exit 1
fi

mkdir -p "${OUT_DIR}"
"${EXECUTABLE}" "${OBJ_FILE}" "${VIEW_DIR}" "${OUT_DIR}"
