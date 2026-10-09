#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/install_dependencies.sh [OPTIONS]

Options:
  --no-sudo  Run the package manager directly (useful in a root container)

Installs the Linux build dependencies used by the CMake project:
CMake, a C++17 compiler, pkg-config, and oneTBB.
USAGE
}

USE_SUDO=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-sudo) USE_SUDO=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if ! command -v sudo >/dev/null 2>&1; then
    USE_SUDO=0
fi
if [[ "${USE_SUDO}" -eq 1 ]] && [[ "$(id -u)" -eq 0 ]]; then
    USE_SUDO=0
fi

run_pkg() {
    if [[ "${USE_SUDO}" -eq 1 ]]; then
        sudo "$@"
    else
        "$@"
    fi
}

if command -v apt-get >/dev/null 2>&1; then
    run_pkg apt-get update
    run_pkg apt-get install -y \
        build-essential cmake pkg-config libtbb-dev unzip
elif command -v dnf >/dev/null 2>&1; then
    run_pkg dnf install -y \
        gcc-c++ make cmake pkgconf tbb-devel unzip
elif command -v zypper >/dev/null 2>&1; then
    run_pkg zypper install -y \
        gcc-c++ make cmake pkg-config tbb-devel unzip
elif command -v pacman >/dev/null 2>&1; then
    run_pkg pacman -Sy --needed --noconfirm \
        base-devel cmake pkgconf onetbb unzip
elif command -v apk >/dev/null 2>&1; then
    run_pkg apk add \
        build-base cmake pkgconfig tbb-dev unzip
else
    echo "Unsupported package manager. Install these dependencies manually:" >&2
    echo "  CMake >= 3.16, C++17 compiler, pkg-config, oneTBB, unzip" >&2
    exit 1
fi

echo
echo "Dependency installation complete."
