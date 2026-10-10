#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PYVISTA_VENV:-${ROOT_DIR}/.venv-pyvista}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'USAGE'
Usage: scripts/install_pyvista_renderer.sh

Creates .venv-pyvista with system-site-packages enabled and installs the
Python dependencies for scripts/render_front_compare.py. The system VTK
installation is reused; PyVista is installed with --no-deps to avoid a second
VTK wheel.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python interpreter not found: ${PYTHON_BIN}" >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    echo "Python 3.10 or newer is required." >&2
    exit 1
fi

if ! "${PYTHON_BIN}" -c 'import vtk' >/dev/null 2>&1; then
    cat >&2 <<'MSG'
System VTK is required but could not be imported.

Install it before rerunning this script, for example:
  sudo apt-get install python3-vtk
or:
  python3 -m pip install vtk
MSG
    exit 1
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    "${PYTHON_BIN}" -m venv --system-site-packages "${VENV_DIR}"
fi

"${VENV_DIR}/bin/python" -m pip install --no-deps pyvista==0.49.0
"${VENV_DIR}/bin/python" -m pip install -r "${ROOT_DIR}/scripts/requirements-pyvista.txt"

"${VENV_DIR}/bin/python" - <<'PY'
import pyvista
import pygltflib
import trimesh
import vtk

print(f"PyVista renderer environment ready: pyvista={pyvista.__version__}, trimesh={trimesh.__version__}, vtk={vtk.vtkVersion.GetVTKVersion()}")
PY

echo
echo "Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
