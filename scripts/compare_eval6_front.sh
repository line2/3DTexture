#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PYVISTA_VENV:-${ROOT_DIR}/.venv-pyvista}"
RENDER_SCRIPT="${ROOT_DIR}/scripts/render_front_compare.py"

usage() {
    cat <<'USAGE'
Usage: scripts/compare_eval6_front.sh DATASET RESULTS [OUTPUT] [extra args...]

DATASET may be eval_6cases.zip or an extracted eval_6cases/ directory.
RESULTS is the algorithm output root containing one directory per case.
OUTPUT defaults to RESULTS/front_compare.

Examples:
  scripts/compare_eval6_front.sh \
      /mnt/c/Users/JdSc-local/Downloads/eval_6cases.zip \
      /home/jdsc-local/tmp/3dtexture-eval6-full

  scripts/compare_eval6_front.sh DATASET RESULTS OUTPUT --case CASE_ID --diff
USAGE
}

if [[ $# -lt 2 ]]; then
    usage >&2
    exit 2
fi

DATASET="$(realpath "$1")"
RESULTS="$(realpath -m "$2")"
shift 2

OUTPUT="${RESULTS}/front_compare"
if [[ $# -gt 0 && "$1" != --* ]]; then
    OUTPUT="$1"
    shift
fi

if [[ ! -e "${DATASET}" ]]; then
    echo "Dataset path not found: ${DATASET}" >&2
    exit 1
fi
if [[ ! -d "${RESULTS}" ]]; then
    echo "Results directory not found: ${RESULTS}" >&2
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${ROOT_DIR}/scripts/install_pyvista_renderer.sh"
fi

RUNNER=("${VENV_DIR}/bin/python" "${RENDER_SCRIPT}")
if [[ -z "${DISPLAY:-}" ]] && command -v xvfb-run >/dev/null 2>&1; then
    RUNNER=(xvfb-run -a "${RUNNER[@]}")
fi

exec "${RUNNER[@]}" \
    --dataset "${DATASET}" \
    --results "${RESULTS}" \
    --output "${OUTPUT}" \
    "$@"
