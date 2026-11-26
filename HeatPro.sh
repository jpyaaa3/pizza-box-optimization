#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_bin=${PYTHON:-python3}
if [[ -z ${PYTHON:-} && -z ${VIRTUAL_ENV:-} && -x "$root/.venv/bin/python" ]]; then
    python_bin="$root/.venv/bin/python"
fi

if (( $# == 0 )); then
    exec "$python_bin" -B "$root/ga_ui.py"
fi
if [[ $1 != --cli ]]; then
    echo 'Usage: ./HeatPro.sh [--cli GA_OPTIONS...]' >&2
    exit 2
fi
shift

for arg in "$@"; do
    case "$arg" in
        --out|--out=*) exec "$python_bin" -B -u "$root/ga.py" "$@" ;;
    esac
done

run_id="$(date +%Y%m%d_%H%M%S)_$$"
exec "$python_bin" -B -u "$root/ga.py" --out "$root/ga_runs/$run_id" "$@"
