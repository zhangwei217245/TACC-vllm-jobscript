#!/usr/bin/env bash

set -Eeuo pipefail

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly HF_CLI="${PROJECT_DIR}/.venv/bin/hf"
readonly MODELS_DIR="${PROJECT_DIR}/models"

log() {
    printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

[[ -x "${HF_CLI}" ]] || {
    printf 'ERROR: %s was not found. Install vLLM in %s/.venv first.\n' \
        "${HF_CLI}" "${PROJECT_DIR}" >&2
    exit 1
}

mkdir -p -- "${MODELS_DIR}"

download_model() {
    local repo_id="$1"
    local directory_name="$2"
    local destination="${MODELS_DIR}/${directory_name}"

    log "Downloading ${repo_id}"
    log "Destination: ${destination}"

    "${HF_CLI}" download "${repo_id}" \
        --local-dir "${destination}"
}

df -h -- "${PROJECT_DIR}"

download_model \
    "Qwen/Qwen3.8-Flash-Next-FP8" \
    "Qwen3.8-Flash-Next-FP8"

download_model \
    "Qwen/Qwen3-Coder-Next-FP8" \
    "Qwen3-Coder-Next-FP8"

log "Both downloads completed"
du -sh -- \
    "${MODELS_DIR}/Qwen3.8-Flash-Next-FP8" \
    "${MODELS_DIR}/Qwen3-Coder-Next-FP8"

