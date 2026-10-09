#!/usr/bin/env bash
set -euo pipefail

BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="${BUILD_DIR:-build/${BUILD_TYPE}}"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cmake -S "${SOURCE_DIR}" -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "${BUILD_DIR}" --parallel

echo
echo "Built: ${BUILD_DIR}/3DTexture"
