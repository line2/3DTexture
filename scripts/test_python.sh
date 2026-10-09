#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec uv run --project "${ROOT_DIR}" --locked \
    python "${ROOT_DIR}/tests/open3d_camera_conversion_test.py"
