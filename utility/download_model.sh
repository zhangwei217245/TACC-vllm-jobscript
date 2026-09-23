#!/usr/bin/env bash

set -Eeuo pipefail

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
readonly HF_CLI="${PROJECT_DIR}/.venv/bin/hf"
readonly MODELS_DIR="${PROJECT_DIR}/models"

usage() {
    printf 'Usage: bash %s [models.txt]\n' "$0"
    printf 'Default list: %s/models/models.txt\n' "${PROJECT_DIR}"
    printf 'One Hugging Face model ID per line; blank lines and # comments are ignored.\n'
}

case ${1:-} in
    -h|--help) usage; exit 0 ;;
esac
[[ $# -le 1 ]] || { usage >&2; exit 2; }
readonly MODEL_LIST="${1:-${PROJECT_DIR}/models/models.txt}"
[[ -f "${MODEL_LIST}" && -r "${MODEL_LIST}" ]] || {
    printf 'ERROR: Model list is not a readable file: %s\n' "${MODEL_LIST}" >&2
    exit 1
}

log() {
    printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

[[ -x "${HF_CLI}" ]] || {
    printf 'ERROR: %s was not found. Run utility/installer.sh first.\n' "${HF_CLI}" >&2
    exit 1
}

mkdir -p -- "${MODELS_DIR}"
df -h -- "${PROJECT_DIR}"

count=0
while IFS= read -r repo_id || [[ -n "$repo_id" ]]; do
    # Trim whitespace (including CRLF carriage returns).
    repo_id="${repo_id#"${repo_id%%[![:space:]]*}"}"
    repo_id="${repo_id%"${repo_id##*[![:space:]]}"}"
    [[ -n "$repo_id" && "$repo_id" != \#* ]] || continue
    [[ "$repo_id" =~ ^[[:alnum:]_][[:alnum:]_.-]*/[[:alnum:]_][[:alnum:]_.-]*$ ]] || {
        printf 'ERROR: Expected organization/model in %s: %s\n' "$MODEL_LIST" "$repo_id" >&2
        exit 1
    }
    destination="${MODELS_DIR}/${repo_id//\//--}"
    log "Downloading ${repo_id}"
    log "Destination: ${destination}"
    # Keep the CLI from consuming subsequent lines of the model list.
    "${HF_CLI}" download "${repo_id}" --local-dir "${destination}" </dev/null
    du -sh -- "${destination}"
    count=$((count + 1))
done < "${MODEL_LIST}"

log "Completed ${count} model download(s)"
