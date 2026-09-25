#!/usr/bin/env bash
# Benchmark an already-running vLLM endpoint; writes one independent run directory.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python=${AIPERF_PYTHON:-$root/.venv-aiperf/bin/python}
[[ -x $python ]] || python=python3
exec "$python" "$root/utility/aiperf_tools.py" benchmark "$@"
