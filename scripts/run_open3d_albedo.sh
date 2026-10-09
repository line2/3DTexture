#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -lt 1 ]]; then
    echo "Usage: scripts/run_open3d_albedo.sh DATASET [extra args...]" >&2
    exit 2
fi
DATASET="$1"
shift
exec uv run --project "${ROOT_DIR}" --locked \
    "${ROOT_DIR}/tools/open3d_project_to_albedo.py" "${DATASET}" "$@"
