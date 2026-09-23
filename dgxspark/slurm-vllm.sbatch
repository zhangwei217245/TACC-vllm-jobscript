#!/usr/bin/env bash
#SBATCH --job-name=vllm
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --output=vllm-run-%j.out
# Four DGX Spark nodes, one GPU each. Add your site's account, partition,
# time limit and GPU request as needed (e.g. sbatch --gres=gpu:1 ... if configured).
# --mem=0 requests all schedulable host memory on each allocated node.
# Submit from the repo root OR dgxspark/. For another working directory, export
# LAUNCHER=/absolute/path/to/repo/dgxspark/launch-vllm.sh or DEPLOY_KIT_ROOT.
# Repo layout: dgxspark/{launch-vllm.sh,slurm-vllm.sbatch},
#             utility/inference-network.sh, vllm_middleware/static_ui.py, ui/chat.html.
# PROJECT, its venv, weights and this kit must be available on every node.
#
# Logs use Slurm's initial working directory (--chdir), captured before any cd:
#   <job-work-dir>/vllm-run-JOBID.out          batch stdout + stderr
#   <job-work-dir>/vllm-run-JOBID/hosts.txt    allocated hosts
#   <job-work-dir>/vllm-run-JOBID/node-*.log   per-node vLLM logs
# Without --chdir, this is the submission directory. The initial directory
# must already exist on the nodes; the script creates only the job subdirectory.
# An explicit --output/--error override can place batch logs somewhere else.
#
# From repo root (absolute script path avoids ambiguity with --chdir):
#   mkdir -p "$PWD/logs"
#   sbatch --chdir="$PWD/logs" "$PWD/dgxspark/slurm-vllm.sbatch" --dry-run
#   sbatch --chdir="$PWD/logs" "$PWD/dgxspark/slurm-vllm.sbatch"
#   CONTEXT_PROFILE=128k SPEC_METHOD=none \
#     sbatch --chdir="$PWD/logs" "$PWD/dgxspark/slurm-vllm.sbatch"
#   MAX_NUM_SEQS=4 VLLM_UI_ENABLE=0 \
#     sbatch --chdir="$PWD/logs" "$PWD/dgxspark/slurm-vllm.sbatch"
#   SERVICE_PORT=8040 VLLM_PUBLIC_BASE_URL=http://dgx.example.org:8040 \
#     sbatch --chdir="$PWD/logs" "$PWD/dgxspark/slurm-vllm.sbatch"
#
# Defaults preserve your requested 1M / 8 sequences / DFlash 15 setup. This is
# an experimental combination, not a verified capacity/performance guarantee.
# Use the non-speculative 128K example above to isolate startup problems.
# The batch allocation stays active while vLLM runs; cancellation/time limit
# stops the serving step. Dry-run also uses an allocation but loads no weights.

set -euo pipefail
die() { printf 'slurm-vllm.sbatch: %s\n' "$*" >&2; exit 1; }
positive_int() {
    local name=$1 value=${!1}
    [[ $value =~ ^[0-9]{1,9}$ ]] || die "$name must be a positive integer."
    (( 10#$value > 0 )) || die "$name must be positive."
    printf -v "$name" '%d' "$((10#$value))"
}
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_JOB_NODELIST:-} ]] || die 'Submit this script with sbatch.'
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || die 'Only --dry-run is supported.'
# Capture the actual initial cwd, including sbatch --chdir, before changing it.
# Keep this absolute path even after the launcher switches to PROJECT.
JOB_WORK_DIR=$(pwd -P)
export JOB_WORK_DIR
# Launcher discovery and caller-relative configuration use the submission dir.
cd -- "${SLURM_SUBMIT_DIR:?}"

# sbatch runs a spooled copy, so BASH_SOURCE is NOT a reliable repo location.
if [[ -z ${LAUNCHER:-} ]]; then
    if [[ -n ${DEPLOY_KIT_ROOT:-} ]]; then
        LAUNCHER=$DEPLOY_KIT_ROOT/dgxspark/launch-vllm.sh
    elif [[ -r $PWD/dgxspark/launch-vllm.sh ]]; then
        LAUNCHER=$PWD/dgxspark/launch-vllm.sh
    else
        LAUNCHER=$PWD/launch-vllm.sh
    fi
fi
[[ $LAUNCHER == /* ]] || LAUNCHER=$PWD/$LAUNCHER
[[ -r $LAUNCHER ]] || die "Launcher missing: $LAUNCHER"
launcher_dir=$(cd -- "$(dirname -- "$LAUNCHER")" && pwd)
LAUNCHER=$launcher_dir/$(basename -- "$LAUNCHER")
DEPLOY_KIT_ROOT=${DEPLOY_KIT_ROOT:-$(cd -- "$launcher_dir/.." && pwd)}
[[ $DEPLOY_KIT_ROOT == /* ]] || DEPLOY_KIT_ROOT=$PWD/$DEPLOY_KIT_ROOT
NETWORK_SCRIPT=${NETWORK_SCRIPT:-$DEPLOY_KIT_ROOT/utility/inference-network.sh}
[[ $NETWORK_SCRIPT == /* ]] || NETWORK_SCRIPT=$PWD/$NETWORK_SCRIPT
[[ -r $NETWORK_SCRIPT ]] || die "Network helper missing: $NETWORK_SCRIPT"
# Resolve PROJECT before srun changes cwd to the job's log directory.
PROJECT=${PROJECT:-$DEPLOY_KIT_ROOT}
[[ -d $PROJECT ]] || die "Project directory missing: $PROJECT"
PROJECT=$(cd -- "$PROJECT" && pwd)
export LAUNCHER NETWORK_SCRIPT DEPLOY_KIT_ROOT PROJECT

run_dir=$JOB_WORK_DIR/vllm-run-$SLURM_JOB_ID
mkdir -p -- "$run_dir"
export HOSTFILE=$run_dir/hosts.txt
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$HOSTFILE"
mapfile -t hosts < "$HOSTFILE"
(( ${#hosts[@]} > 0 )) || die 'Slurm returned an empty host list.'
[[ -n ${SLURM_JOB_NUM_NODES:-} ]] || die 'Missing SLURM_JOB_NUM_NODES.'
positive_int SLURM_JOB_NUM_NODES
(( ${#hosts[@]} == SLURM_JOB_NUM_NODES )) || die 'Host list does not match allocation size.'
head_node=${hosts[0]}
export NUM_NODES=${#hosts[@]}

# Export the batch job's current environment to srun tasks. This cannot restore
# submission-shell variables discarded by sbatch --export=NONE; use --export=ALL
# (the usual default) when passing settings via VAR=value sbatch ... .
export SLURM_EXPORT_ENV=ALL
export NET_TRANSPORT=${NET_TRANSPORT:-rdma}
export MASTER_PORT=${MASTER_PORT:-8041}
# One GPU per node: tensor-parallel across all nodes, one pipeline stage.
export TP_SIZE=${TP_SIZE:-$NUM_NODES}
export PP_SIZE=${PP_SIZE:-1}
positive_int TP_SIZE
positive_int PP_SIZE
(( TP_SIZE * PP_SIZE == NUM_NODES )) || die 'For this one-GPU-per-node job, TP_SIZE * PP_SIZE must equal NUM_NODES.'
export PYTHONUNBUFFERED=1
unset NODE_RANK LOCAL_NODE_NAME

# Defaults apply only when a submission-time value is absent/empty.
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
export CONTEXT_PROFILE=${CONTEXT_PROFILE:-1m}
export SPEC_METHOD=${SPEC_METHOD:-dflash}
if [[ $SPEC_METHOD == dflash ]]; then
    export SPEC_TOKENS=${SPEC_TOKENS:-15}
    export SPEC_MODEL=${SPEC_MODEL:-$PROJECT/models/Qwen3-Coder-Next-DFlash}
fi
if [[ $SPEC_METHOD != none ]] && (( PP_SIZE != 1 )); then
    die 'The launcher speculative presets require PP_SIZE=1; use TP_SIZE=NUM_NODES.'
fi
# UI settings are frontend-only; workers never load the middleware or UI files.
export VLLM_UI_ENABLE=${VLLM_UI_ENABLE:-1}
export VLLM_MIDDLEWARE_DIR=${VLLM_MIDDLEWARE_DIR:-$DEPLOY_KIT_ROOT/vllm_middleware}
export VLLM_UI_DIR=${VLLM_UI_DIR:-$DEPLOY_KIT_ROOT/ui}
export VLLM_UI_PAGE=${VLLM_UI_PAGE:-chat.html}
export VLLM_PUBLIC_BASE_URL=${VLLM_PUBLIC_BASE_URL:-}
# Empty public URL => browser same origin, including external port or SSH tunnel.
# Explicit public URL is the server root WITHOUT /v1; it does not change binding.

if [[ -z ${HEAD_IP:-} ]]; then
    # Finish network detection on the first allocated node before starting all
    # engines. Diagnostics go to stderr; stdout must contain only one IPv4 IP.
    HEAD_IP=$(srun --mpi=none --nodes=1 --ntasks=1 --ntasks-per-node=1 \
        --nodelist="$head_node" --export=ALL bash -c '
        set -euo pipefail
        args=(--transport "$NET_TRANSPORT")
        [[ -z "${NET_IFACE:-}" ]] || args+=(--iface "$NET_IFACE")
        [[ -z "${NET_HCA:-}" ]] || args+=(--hca "$NET_HCA")
        [[ -z "${NET_LOCAL_IP:-}" ]] || args+=(--local-ip "$NET_LOCAL_IP")
        source "$NETWORK_SCRIPT" "${args[@]}" >&2 || exit 1
        printf "%s\n" "$INFER_LOCAL_IP"
    ') || die 'Could not detect the first node network IP.'
fi
# Full address/bind validation also runs in launch-vllm.sh on the head.
[[ $HEAD_IP =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || die 'HEAD_IP must be one IPv4 address.'
export HEAD_IP
printf 'Job %s: head=%s address=%s:%s TP=%s PP=%s\n' \
    "$SLURM_JOB_ID" "$head_node" "$HEAD_IP" "$MASTER_PORT" "$TP_SIZE" "$PP_SIZE"
printf 'Job working directory (logs): %s\nSubmission directory (launcher discovery): %s\n' \
    "$JOB_WORK_DIR" "$SLURM_SUBMIT_DIR"
printf 'Host file: %s\nPer-node logs: %s/node-<hostname>.log\n' "$HOSTFILE" "$run_dir"
printf 'Context=%s; sequences=%s; batch tokens=%s; speculation=%s; UI=%s\n' \
    "$CONTEXT_PROFILE" "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" "$SPEC_METHOD" "$VLLM_UI_ENABLE"
cat -- "$HOSTFILE"

# One task per node supervises vLLM; vLLM starts its own GPU processes.
# --mpi=none avoids MPI integration; --kill-on-bad-exit stops peers on failure.
srun --mpi=none --nodes="$NUM_NODES" --ntasks="$NUM_NODES" \
    --ntasks-per-node=1 --kill-on-bad-exit=1 --export=ALL \
    --output="$run_dir/node-%N.log" --error="$run_dir/node-%N.log" \
    bash -c '
        set -euo pipefail
        export LOCAL_NODE_NAME="${SLURMD_NODENAME:?Missing Slurm node identity}"
        exec bash "$LAUNCHER" "$@"
    ' vllm-node "$@"
