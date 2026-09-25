#!/usr/bin/env bash

set -Eeuo pipefail

readonly PYTHON_VERSION="${PYTHON_VERSION:-3.14}"
readonly VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
readonly AIPERF_VERSION="${AIPERF_VERSION:-0.13.0}"
readonly AIPERF_PYTHON_VERSION="${AIPERF_PYTHON_VERSION:-3.13}"
readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly VENV_DIR="${PROJECT_DIR}/.venv"
readonly MODELS_DIR="${PROJECT_DIR}/models"
readonly AIPERF_VENV_DIR="${PROJECT_DIR}/.venv-aiperf"

log() {
    printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ "$(uname -s)" == "Linux" ]] || die "This installer requires Linux."

if [[ "$(uname -m)" != "aarch64" ]]; then
    printf 'WARNING: DGX Spark should report aarch64; detected %s.\n' "$(uname -m)" >&2
fi

[[ "${PYTHON_VERSION}" != 3.15* ]] || \
    die "vLLM ${VLLM_VERSION} requires Python < 3.15; use Python 3.14."

mkdir -p -- "${MODELS_DIR}" "${PROJECT_DIR}/logs" "${PROJECT_DIR}/aiperf-out"

if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
else
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh

    UV_BIN="${HOME}/.local/bin/uv"
    if [[ ! -x "${UV_BIN}" && -x "${HOME}/.cargo/bin/uv" ]]; then
        UV_BIN="${HOME}/.cargo/bin/uv"
    fi
    [[ -x "${UV_BIN}" ]] || die "uv installation completed, but the executable was not found."
fi

log "Using $(${UV_BIN} --version)"
log "Installing managed Python ${PYTHON_VERSION}"
"${UV_BIN}" python install "${PYTHON_VERSION}"

if [[ -d "${VENV_DIR}" ]]; then
    [[ -x "${VENV_DIR}/bin/python" ]] || \
        die "${VENV_DIR} exists but is not a valid virtual environment."

    existing_python_version="$("${VENV_DIR}/bin/python" -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "${existing_python_version}" == "${PYTHON_VERSION}" ]] || \
        die "${VENV_DIR} uses Python ${existing_python_version}; expected ${PYTHON_VERSION}."

    log "Reusing ${VENV_DIR}"
else
    log "Creating ${VENV_DIR}"
    "${UV_BIN}" venv \
        --python "${PYTHON_VERSION}" \
        --managed-python \
        --seed \
        "${VENV_DIR}"
fi

log "Installing base vLLM ${VLLM_VERSION}"
"${UV_BIN}" pip install \
    --python "${VENV_DIR}/bin/python" \
    --upgrade \
    --torch-backend=auto \
    "vllm==${VLLM_VERSION}"

# AIPerf 0.13 requires Python >=3.11,<3.14. Keep its dependencies and
# interpreter separate from the Python 3.14 serving environment.
case "$AIPERF_PYTHON_VERSION" in 3.11|3.12|3.13) ;; *) die 'AIPerf requires Python 3.11, 3.12, or 3.13.' ;; esac
log "Installing AIPerf ${AIPERF_VERSION} in ${AIPERF_VENV_DIR}"
"${UV_BIN}" python install "${AIPERF_PYTHON_VERSION}"
if [[ -d $AIPERF_VENV_DIR ]]; then
    [[ -x $AIPERF_VENV_DIR/bin/python ]] || die "Invalid environment: $AIPERF_VENV_DIR"
    [[ $("$AIPERF_VENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")') == "$AIPERF_PYTHON_VERSION" ]] || die 'Existing AIPerf environment uses a different Python version.'
else
    "$UV_BIN" venv --python "$AIPERF_PYTHON_VERSION" --managed-python "$AIPERF_VENV_DIR"
fi
"$UV_BIN" pip install --python "$AIPERF_VENV_DIR/bin/python" --upgrade "aiperf==$AIPERF_VERSION"
"$AIPERF_VENV_DIR/bin/aiperf" --version

log "Verifying vLLM, PyTorch, and CUDA"
"${VENV_DIR}/bin/python" <<'PY'
import platform
import torch
import vllm

print(f"Python:         {platform.python_version()}")
print(f"vLLM:           {vllm.__version__}")
print(f"PyTorch:        {torch.__version__}")
print(f"CUDA runtime:   {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise SystemExit("Installation completed, but PyTorch cannot see the GPU.")

print(f"GPU:            {torch.cuda.get_device_name(0)}")
PY

log "Installation completed successfully"
printf 'Activate: source %q\n' "${VENV_DIR}/bin/activate"
printf 'Models:   %s\n' "${MODELS_DIR}"
printf 'AIPerf:   %s\n' "${AIPERF_VENV_DIR}/bin/aiperf"
printf 'Benchmark: bash %q --help\n' "${PROJECT_DIR}/utility/benchmark_aiperf.sh"
