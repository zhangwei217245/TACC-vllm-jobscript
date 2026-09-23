#!/usr/bin/env bash
#SBATCH --job-name=vllm
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=vllm-%j.out

# Four DGX Spark nodes, one GPU each. Adjust resource/account/partition options
# for the site. --mem=0 requests all schedulable host memory on each node.
# Submit from a shared directory containing launch-vllm.sh and inference-network.sh.
# PROJECT, its virtual environment and model files must exist at the same paths
# on all nodes (shared storage or identical local installations).
#
#   sbatch launch-vllm.sbatch --dry-run
#   sbatch launch-vllm.sbatch
#   TP_SIZE=1 PP_SIZE=4 sbatch launch-vllm.sbatch  # only for a PP-capable model
#
# Job-specific hosts/logs are under vllm-JOBID/ in the submission directory.
# HEAD_IP is detected on the first allocated node unless explicitly exported.
# Network selection is a local heuristic; it does not certify peer connectivity.


set -euo pipefail
die() { printf 'launch-vllm.sbatch: %s\n' "$*" >&2; exit 1; }
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || die 'Submit this script with sbatch.'
[[ $# == 0 || ( $# == 1 && "$1" == --dry-run ) ]] || die 'Only --dry-run is supported.'
cd -- "${SLURM_SUBMIT_DIR:?}"
export LAUNCHER=${LAUNCHER:-$SLURM_SUBMIT_DIR/launch-vllm.sh}
export NETWORK_SCRIPT=${NETWORK_SCRIPT:-$SLURM_SUBMIT_DIR/inference-network.sh}
[[ "$LAUNCHER" == /* ]] || LAUNCHER="$PWD/$LAUNCHER"
[[ "$NETWORK_SCRIPT" == /* ]] || NETWORK_SCRIPT="$PWD/$NETWORK_SCRIPT"
[[ -r "$LAUNCHER" && -r "$NETWORK_SCRIPT" ]] || die 'Launcher or network helper missing.'

run_dir="$SLURM_SUBMIT_DIR/vllm-$SLURM_JOB_ID"
mkdir -p -- "$run_dir"
export HOSTFILE="$run_dir/hosts.txt"
scontrol show hostnames "$SLURM_JOB_NODELIST" > "$HOSTFILE"
mapfile -t hosts < "$HOSTFILE"
(( ${#hosts[@]} > 0 )) || die 'Slurm returned an empty host list.'
(( ${#hosts[@]} == SLURM_JOB_NUM_NODES )) || die 'Host list does not match allocation size.'
head_node=${hosts[0]}
export NUM_NODES=${#hosts[@]}
# SLURM_EXPORT_ENV also supports jobs submitted with --export=NONE.
export SLURM_EXPORT_ENV=ALL
export NET_TRANSPORT=${NET_TRANSPORT:-rdma}
export MASTER_PORT=${MASTER_PORT:-8041}
# Defaults for the requested one-GPU-per-node allocation: TP across nodes.
export TP_SIZE=${TP_SIZE:-1}
export PP_SIZE=${PP_SIZE:-$NUM_NODES}
export PYTHONUNBUFFERED=1
# Each task gets its identity from Slurm and derives its rank from HOSTFILE.
unset NODE_RANK LOCAL_NODE_NAME

if [[ -z "${HEAD_IP:-}" ]]; then
    # Complete this one-node step before launching the serving step. Keep
    # diagnostics on stderr; capture only the selected IPv4 address on stdout.
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
[[ "$HEAD_IP" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || die 'HEAD_IP must be one IPv4 address.'
export HEAD_IP
printf 'Job %s: head=%s address=%s:%s TP=%s PP=%s\n' \
    "$SLURM_JOB_ID" "$head_node" "$HEAD_IP" "$MASTER_PORT" "$TP_SIZE" "$PP_SIZE"
printf 'Host file: %s\nPer-node logs: %s/node-<hostname>.log\n' "$HOSTFILE" "$run_dir"
cat -- "$HOSTFILE"

# Slurm launches one supervisor per node. vLLM starts its GPU workers itself;
# Open MPI is not used. A task failure terminates the other tasks in this step.
srun --mpi=none --nodes="$NUM_NODES" --ntasks="$NUM_NODES" \
    --ntasks-per-node=1 --kill-on-bad-exit=1 --export=ALL \
    --output="$run_dir/node-%N.log" --error="$run_dir/node-%N.log" \
    bash -c '
        set -euo pipefail
        export LOCAL_NODE_NAME="${SLURMD_NODENAME:?Missing Slurm node identity}"
        exec bash "$LAUNCHER" "$@"
    ' vllm-node "$@"
