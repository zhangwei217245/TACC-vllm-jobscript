#!/usr/bin/env bash
# Creates the output directory before Slurm opens batch stdout/stderr.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export DEPLOY_KIT_ROOT=${DEPLOY_KIT_ROOT:-$root}
export LOG_DIR=${LOG_DIR:-$DEPLOY_KIT_ROOT/logs}
mkdir -p -- "$LOG_DIR"
LOG_DIR=$(cd -- "$LOG_DIR" && pwd)
export LOG_DIR
slurm_args=() job_args=()
for arg in "$@"; do
    case $arg in
        --help|-h) printf 'Usage: bash dgxspark/submit-vllm.sh [sbatch options] [--dry-run]\nLogs default to <repo>/logs. --dry-run runs launcher preflight inside an allocation.\n'; exit 0 ;;
        --dry-run) job_args+=("$arg") ;;
        *) slurm_args+=("$arg") ;;
    esac
done
exec sbatch --export=ALL --chdir="$DEPLOY_KIT_ROOT" \
    --output="$LOG_DIR/vllm-run-%j.out" --error="$LOG_DIR/vllm-run-%j.out" \
    ${slurm_args[@]+"${slurm_args[@]}"} "$DEPLOY_KIT_ROOT/dgxspark/slurm-vllm.sbatch" ${job_args[@]+"${job_args[@]}"}
