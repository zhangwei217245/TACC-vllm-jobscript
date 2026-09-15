#!/usr/bin/env bash

set -Eeuo pipefail

readonly PYTHON_VERSION="${PYTHON_VERSION:-3.14}"
readonly VLLM_VERSION="${VLLM_VERSION:-0.28.0}"
readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly VENV_DIR="${PROJECT_DIR}/.venv"
readonly MODELS_DIR="${PROJECT_DIR}/models"

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

mkdir -p -- "${MODELS_DIR}"

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

