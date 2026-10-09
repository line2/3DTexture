#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build/Release}"
EXECUTABLE="${BUILD_DIR}/3DTexture"

usage() {
    cat <<'USAGE'
Usage: scripts/run_eval_6cases.sh ZIP_FILE OUTPUT_DIR [extra 3DTexture args...]

Extracts eval_6cases.zip to a temporary directory, runs all cases, and writes
a per-case output directory plus a top-level summary.json.
USAGE
}

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi

ZIP_FILE="$(realpath "$1")"
OUTPUT_DIR="$(realpath -m "$2")"
shift 2

if [[ ! -f "${ZIP_FILE}" ]]; then
    echo "ZIP file not found: ${ZIP_FILE}" >&2
    exit 1
fi
if ! command -v unzip >/dev/null 2>&1; then
    echo "unzip is required. Install it or rerun scripts/install_dependencies.sh." >&2
    exit 1
fi
if [[ ! -x "${EXECUTABLE}" ]]; then
    "${ROOT_DIR}/scripts/build.sh"
fi
if [[ ! -x "${EXECUTABLE}" ]]; then
    echo "Executable not found: ${EXECUTABLE}" >&2
    exit 1
fi

EXTRACT_DIR="$(mktemp -d /tmp/3dtexture-eval6cases.XXXXXX)"
cleanup() {
    rm -rf "${EXTRACT_DIR}"
}
trap cleanup EXIT

unzip -q "${ZIP_FILE}" -d "${EXTRACT_DIR}"
DATASET_ROOT="${EXTRACT_DIR}/eval_6cases"
if [[ ! -d "${DATASET_ROOT}" ]]; then
    echo "Expected eval_6cases/ root inside ${ZIP_FILE}" >&2
    exit 1
fi

"${EXECUTABLE}" run \
    --dataset "${DATASET_ROOT}" \
    --output "${OUTPUT_DIR}" \
    --format eval6cases \
    "$@"
