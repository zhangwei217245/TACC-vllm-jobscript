#!/usr/bin/env bash
#SBATCH --job-name=vllm
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --output=logs/vllm-run-%j.out
# Four DGX Spark nodes, one GPU each. Add your site's account, partition,
# time limit and GPU request as needed (e.g. sbatch --gres=gpu:1 ... if configured).
# --mem=0 requests all schedulable host memory on each allocated node.
# Submit through dgxspark/submit-vllm.sh from any directory. For raw sbatch,
# submit from the repo root with logs/ already present. For another directory, export
# LAUNCHER=/absolute/path/to/repo/dgxspark/launch-vllm.sh or DEPLOY_KIT_ROOT.
# Repo layout: dgxspark/{launch-vllm.sh,slurm-vllm.sbatch},
#             utility/inference-network.sh, vllm_middleware/static_ui.py, ui/chat.html.
# PROJECT, its venv, weights and this kit must be available on every node.
#
# Logs default to <repo>/logs, independent of the submission directory:
#   logs/vllm-run-JOBID.out          batch stdout + stderr
#   logs/vllm-run-JOBID/hosts.txt    allocated hosts
#   logs/vllm-run-JOBID/node-*.log   per-node vLLM logs
# The submission helper creates logs/ before sbatch opens stdout. LOG_DIR can
# override this root; with raw sbatch also pass matching --output/--error paths.
#
# From repo root (absolute script path avoids ambiguity with --chdir):
#   bash dgxspark/submit-vllm.sh --dry-run
#   bash dgxspark/submit-vllm.sh
#   CONTEXT_PROFILE=128k SPEC_METHOD=none \
#     bash dgxspark/submit-vllm.sh
#   MAX_NUM_SEQS=4 VLLM_UI_ENABLE=0 \
#     bash dgxspark/submit-vllm.sh
#   SERVICE_PORT=8040 VLLM_PUBLIC_BASE_URL=http://dgx.example.org:8040 \
#     bash dgxspark/submit-vllm.sh
#
# Defaults: target TP=2/PP=2, speculation off, 1M context and 8 sequences.
# The topology and speculative method match launch-vllm.sh; batch concurrency
# is explicitly higher. Extended context remains experimental.
# Use the non-speculative 128K example above to isolate startup problems.
# The batch allocation stays active while vLLM runs; cancellation/time limit
# stops the serving step. Dry-run also uses an allocation but loads no weights.

set -euo pipefail
(( BASH_VERSINFO[0] >= 4 )) || { printf 'Bash 4 or later is required.\n' >&2; exit 1; }
die() { printf 'slurm-vllm.sbatch: %s\n' "$*" >&2; exit 1; }
positive_int() {
    local name=$1 value=${!1}
    [[ $value =~ ^[0-9]{1,9}$ ]] || die "$name must be a positive integer."
    (( 10#$value > 0 )) || die "$name must be positive."
    printf -v "$name" '%d' "$((10#$value))"
}
[[ -n ${SLURM_JOB_ID:-} && -n ${SLURM_JOB_NODELIST:-} ]] || die 'Submit this script with sbatch.'
[[ $# == 0 || ( $# == 1 && $1 == --dry-run ) ]] || die 'Only --dry-run is supported.'
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
LOG_DIR=${LOG_DIR:-$DEPLOY_KIT_ROOT/logs}
[[ $LOG_DIR == /* ]] || LOG_DIR=$PWD/$LOG_DIR
mkdir -p -- "$LOG_DIR"
JOB_WORK_DIR=$(cd -- "$LOG_DIR" && pwd)
export LOG_DIR JOB_WORK_DIR

# First two model IDs select the target and optional auxiliary checkpoint.
# Explicit submission-time model overrides still take precedence.
MODEL_REPO=${MODEL_REPO:-$PROJECT/models}
[[ $MODEL_REPO == /* ]] || MODEL_REPO=$PWD/$MODEL_REPO
MODEL_LIST=${MODEL_LIST:-$PROJECT/models/models.txt}
[[ $MODEL_LIST == /* ]] || MODEL_LIST=$PWD/$MODEL_LIST
[[ -f $MODEL_LIST && -r $MODEL_LIST ]] || die "Model list missing or unreadable: $MODEL_LIST"
model_ids=()
while IFS= read -r repo_id || [[ -n $repo_id ]]; do
    repo_id="${repo_id#"${repo_id%%[![:space:]]*}"}"
    repo_id="${repo_id%"${repo_id##*[![:space:]]}"}"
    [[ -n $repo_id && $repo_id != \#* ]] || continue
    [[ $repo_id =~ ^[[:alnum:]_][[:alnum:]_.-]*/[[:alnum:]_][[:alnum:]_.-]*$ ]] ||
        die "Expected organization/model in $MODEL_LIST: $repo_id"
    model_ids+=("$repo_id")
done < "$MODEL_LIST"
(( ${#model_ids[@]} > 0 )) || die "Model list has no primary model: $MODEL_LIST"
MODEL_NAME=${MODEL_NAME:-${model_ids[0]//\//--}}
MODEL_PATH=${MODEL_PATH:-$MODEL_REPO/$MODEL_NAME}
[[ $MODEL_PATH == /* ]] || MODEL_PATH=$PWD/$MODEL_PATH
export MODEL_LIST MODEL_REPO MODEL_NAME MODEL_PATH

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
# Match the launcher's four-GPU baseline. Other allocations need explicit TP/PP.
export TP_SIZE=${TP_SIZE:-2}
export PP_SIZE=${PP_SIZE:-2}
positive_int TP_SIZE
positive_int PP_SIZE
(( TP_SIZE * PP_SIZE == NUM_NODES )) || die "For this one-GPU-per-node job, TP_SIZE=$TP_SIZE * PP_SIZE=$PP_SIZE must equal NUM_NODES=$NUM_NODES. Set both TP_SIZE and PP_SIZE for a different topology."
export PYTHONUNBUFFERED=1
unset NODE_RANK LOCAL_NODE_NAME

# Defaults apply only when a submission-time value is absent/empty.
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
export CONTEXT_PROFILE=${CONTEXT_PROFILE:-1m}
export SPEC_METHOD=${SPEC_METHOD:-none}
case $SPEC_METHOD in
    dflash|eagle3)
        if [[ -z ${SPEC_MODEL:-} ]]; then
            (( ${#model_ids[@]} >= 2 )) ||
                die "SPEC_METHOD=$SPEC_METHOD requires a second model in $MODEL_LIST or an explicit SPEC_MODEL."
            SPEC_MODEL=$MODEL_REPO/${model_ids[1]//\//--}
        fi
        export SPEC_MODEL
        if [[ $SPEC_METHOD == dflash ]]; then
            export SPEC_TOKENS=${SPEC_TOKENS:-15}
        fi ;;
    none)
        [[ -z ${SPEC_MODEL:-} && -z ${SPEC_TOKENS:-} ]] ||
            die 'SPEC_METHOD=none requires SPEC_MODEL and SPEC_TOKENS to be unset.' ;;
    ngram)
        (( PP_SIZE == 1 )) || die 'N-gram preset requires PP_SIZE=1; set TP_SIZE to the allocated GPU count.'
        [[ -z ${SPEC_MODEL:-} ]] || die 'N-gram drafting does not use SPEC_MODEL.'
        [[ ${VLLM_USE_V2_MODEL_RUNNER:-0} != 1 ]] || die 'N-gram preset does not support VLLM_USE_V2_MODEL_RUNNER=1.' ;;
    *) die 'SPEC_METHOD must be none, ngram, dflash, or eagle3.' ;;
esac
# Draft PP is fixed to 1 by compatible vLLM builds, not a CLI option.
# V2 DFlash uses the target TP group, including on the last PP stage.
if [[ ${TACC_QWEN3NEXT_PP_DFLASH:-0} == 1 ]]; then
    export SPEC_TP_SIZE=${SPEC_TP_SIZE:-$TP_SIZE}
else
    export SPEC_TP_SIZE=${SPEC_TP_SIZE:-1}
fi
# UI settings are frontend-only; workers never load the middleware or UI files.
export VLLM_UI_ENABLE=${VLLM_UI_ENABLE:-1}
export VLLM_MIDDLEWARE_DIR=${VLLM_MIDDLEWARE_DIR:-$DEPLOY_KIT_ROOT/vllm_middleware}
export VLLM_UI_DIR=${VLLM_UI_DIR:-$DEPLOY_KIT_ROOT/ui}
export VLLM_UI_PAGE=${VLLM_UI_PAGE:-chat.html}
export VLLM_PUBLIC_BASE_URL=${VLLM_PUBLIC_BASE_URL:-}
# Resolve caller-relative UI paths before the serving step changes directory.
for path_name in VLLM_MIDDLEWARE_DIR VLLM_UI_DIR; do
    [[ ${!path_name} == /* ]] || printf -v "$path_name" '%s/%s' "$PWD" "${!path_name}"
done
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
    --chdir="$JOB_WORK_DIR" \
    --output="$run_dir/node-%N.log" --error="$run_dir/node-%N.log" \
    bash -c '
        set -euo pipefail
        export LOCAL_NODE_NAME="${SLURMD_NODENAME:?Missing Slurm node identity}"
        exec bash "$LAUNCHER" "$@"
    ' vllm-node "$@"
