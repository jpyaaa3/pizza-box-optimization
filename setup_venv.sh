#!/usr/bin/env bash
# Set up the Linux environment used by HeatPro.sh.
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-python3}
if ! "$python_bin" -c 'import sys; sys.exit(sys.version_info < (3, 10))'; then
    echo 'HeatPro requires Python 3.10 or later. Set PYTHON to its executable.' >&2
    exit 1
fi

packages=()
"$python_bin" -c 'import ensurepip, venv' 2>/dev/null || packages+=(python3-venv)
"$python_bin" -c 'import tkinter' 2>/dev/null || packages+=(python3-tk)
"$python_bin" -c 'from ctypes import CDLL; CDLL("libGLU.so.1")' 2>/dev/null || packages+=(libglu1-mesa)
if (( ${#packages[@]} )); then
    if ! command -v apt-get >/dev/null; then
        echo "Install the system equivalents of: ${packages[*]}, then rerun setup." >&2
        exit 1
    fi
    as_root=()
    if (( EUID != 0 )); then as_root=(sudo); fi
    "${as_root[@]}" apt-get update
    "${as_root[@]}" apt-get install -y "${packages[@]}"
fi

env_dir="$root/.venv"
if [[ ! -x "$env_dir/bin/python" ]]; then
    "$python_bin" -m venv "$env_dir"
fi
"$env_dir/bin/python" -m pip install -r "$root/requirements.txt"
"$env_dir/bin/python" - <<'PY'
import gmsh
import matplotlib
import numpy
import scipy
import tkinter

gmsh.initialize([], readConfigFiles=False)
gmsh.finalize()
print("HeatPro environment ready.")
PY
