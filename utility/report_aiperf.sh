#!/usr/bin/env bash
# Scan recorded results only; no AIPerf installation or running server is needed.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${AIPERF_PYTHON:-$root/.venv-aiperf/bin/python}
[[ -x $python ]] || python=python3
exec "$python" "$root/utility/aiperf_tools.py" report "$@"
