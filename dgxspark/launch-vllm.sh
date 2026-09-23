#!/usr/bin/env bash
# Run this launcher on each node; put inference-network.sh and hosts.txt beside it.
# hosts.txt contains one hostname per line in the same order on every node.
# Its first host is rank 0; subsequent hosts are ranks 1, 2, etc.
# The network helper configures each node locally; it does not elect the head.
# All nodes MUST share the host list, HEAD_IP, MASTER_PORT, TP_SIZE, PP_SIZE and
# model settings. HEAD_IP must belong to the first listed host.
# NODE_RANK and NUM_NODES are derived from the file. No remote processes launch.
# Defaults: TP=GPUs/node, PP=number of hosts,
# Qwen3-Coder-Next-FP8, 256K context, memory fraction 0.6, instanttensor.
# These defaults do not certify model PP support or available cache capacity.
#
# Examples:
#   bash ./launch-vllm.sh --dry-run
#   scontrol show hostnames "$SLURM_JOB_NODELIST" > hosts.txt  # once per allocation
#   HOSTFILE=/shared/job-hosts.txt bash ./launch-vllm.sh
#   NET_TRANSPORT=socket MAX_MODEL_LEN=32768 bash ./launch-vllm.sh
#   TP_SIZE=4 PP_SIZE=1 bash ./launch-vllm.sh
#
# --dry-run performs local network/port/GPU checks and prints the command;
# it does not load weights or start vLLM. Checks do not reserve ports.
set -euo pipefail

die() { printf 'launch-vllm: %s\n' "$*" >&2; exit 1; }
dry_run=0
case "${1:-}" in
    --dry-run) dry_run=1; shift;;
    -h|--help)
        cat <<'HELP'
Usage: bash launch-vllm.sh [--dry-run]

Configure via environment variables (defaults are in this script):
  PROJECT, NETWORK_SCRIPT, HEAD_IP, MASTER_PORT
  HOSTFILE (default: hosts.txt beside this launcher), LOCAL_NODE_NAME
  MODEL_NAME, MODEL_REPO, MODEL_PATH, MAX_MODEL_LEN, GPU_MEM_UTILIZATION
  MAX_NUM_SEQS, LOAD_FORMAT, TP_SIZE, PP_SIZE
  NET_TRANSPORT=rdma|auto|socket, NET_DEBUG=1|0, NET_IFACE, NET_HCA
  NET_LOCAL_IP, NET_USE_MASTER_ROUTE=1|0 (default 0: automatic local NIC ranking)
  HTTP_PORTS (default 8040-8050), SERVICE_PORT (optional exact port override)
  HTTP_IFACE, HTTP_CLIENT, HTTP_IP, HTTP_AUDIT=1|0
  HTTP_RESERVED_PORTS (extra exclusions; MASTER_PORT is always excluded)
  MOE_BACKEND (optional; passed on all nodes only when nonempty)
  ENFORCE_EAGER=1|0 (default 0)

Only rank 0 uses HTTP detection. Workers get --headless and no HTTP options.
The host file is required: one hostname per line, in identical order everywhere.
Blank lines and full-line # comments are ignored. Short names and FQDNs match
case-insensitively by their first label; these labels must be unique in the file.
LOCAL_NODE_NAME overrides hostname -s for sites using Slurm node aliases.
NODE_RANK and NUM_NODES come from the file; conflicting environment values fail.
HEAD_IP remains shared configuration and must address the FIRST listed host.
Homogeneous GPU counts assumed.
HTTP bind availability does not prove firewall permission/client reachability.
HELP
        exit 0;;
esac
(( $# == 0 )) || die 'Unknown argument; use --help.'

LAUNCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-/opt/share/gits/Agentic/vllm}
[[ -d "$PROJECT" ]] || die "Project directory missing: $PROJECT"
PROJECT=$(cd -- "$PROJECT" && pwd)
NETWORK_SCRIPT=${NETWORK_SCRIPT:-$LAUNCH_DIR/inference-network.sh}
# Resolve a relative override before changing into the project directory.
[[ "$NETWORK_SCRIPT" == /* ]] || NETWORK_SCRIPT="$PWD/$NETWORK_SCRIPT"
HEAD_IP=${HEAD_IP:-192.168.1.1}
MASTER_PORT=${MASTER_PORT:-8041}
MODEL_NAME=${MODEL_NAME:-Qwen3-Coder-Next-FP8}
MODEL_REPO=${MODEL_REPO:-$PROJECT/models}
MODEL_PATH=${MODEL_PATH:-$MODEL_REPO/$MODEL_NAME}
[[ "$MODEL_PATH" == /* ]] || MODEL_PATH="$PWD/$MODEL_PATH"
MAX_MODEL_LEN=${MAX_MODEL_LEN:-262144}
GPU_MEM_UTILIZATION=${GPU_MEM_UTILIZATION:-0.6}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
LOAD_FORMAT=${LOAD_FORMAT:-instanttensor}
NET_TRANSPORT=${NET_TRANSPORT:-rdma}
HTTP_PORTS=${HTTP_PORTS:-8040-8050}

# The ordered host file is the authority for membership and rank on every run.
HOSTFILE=${HOSTFILE:-$LAUNCH_DIR/hosts.txt}
[[ "$HOSTFILE" == /* ]] || HOSTFILE="$PWD/$HOSTFILE"
[[ -f "$HOSTFILE" && -r "$HOSTFILE" ]] || die "Host file missing or unreadable: $HOSTFILE"
node_name=${LOCAL_NODE_NAME:-$(hostname -s)}
local_key=${node_name,,}
local_key=${local_key%%.*}
[[ -n "$local_key" ]] || die 'Cannot determine the local hostname.'
declare -a NODES=()
declare -A seen_hosts=()
detected_rank=-1
line_number=0
while IFS= read -r host_line || [[ -n "$host_line" ]]; do
    line_number=$((line_number + 1))
    host_line=${host_line%$'\r'}
    # read trims surrounding spaces/tabs; additional columns are not permitted.
    read -r host_entry extra <<< "$host_line"
    [[ -n "$host_entry" && "$host_entry" != \#* ]] || continue
    [[ -z "$extra" && "$host_entry" =~ ^[[:alnum:]_][[:alnum:]_.-]*$ ]] ||
        die "Invalid host at $HOSTFILE:$line_number; use one expanded hostname per line."
    host_key=${host_entry,,}
    host_key=${host_key%%.*}
    [[ -z "${seen_hosts[$host_key]+present}" ]] ||
        die "Duplicate or ambiguous short hostname '$host_key' in $HOSTFILE."
    seen_hosts[$host_key]=1
    [[ "$host_key" != "$local_key" ]] || detected_rank=${#NODES[@]}
    NODES+=("$host_entry")
done < "$HOSTFILE"
(( ${#NODES[@]} > 0 )) || die "Host file is empty: $HOSTFILE"
(( detected_rank >= 0 )) || die "Local host '$node_name' is absent from $HOSTFILE; use LOCAL_NODE_NAME for a Slurm alias."
# Retained environment values may confirm the file, but cannot override it.
for derived_name in NODE_RANK NUM_NODES; do
    supplied_value=${!derived_name-}
    expected_value=$detected_rank
    [[ "$derived_name" != NUM_NODES ]] || expected_value=${#NODES[@]}
    if [[ -n "$supplied_value" ]]; then
        [[ "$supplied_value" =~ ^[0-9]{1,6}$ ]] || die "$derived_name must be an integer."
        (( 10#$supplied_value == expected_value )) || die "$derived_name=$supplied_value conflicts with host file value $expected_value."
    fi
done
NODE_RANK=$detected_rank
NUM_NODES=${#NODES[@]}
HEAD_NODE=${NODES[0]}
printf 'Host file: %s; local=%s; NODE_RANK=%s; NUM_NODES=%s; HEAD_NODE=%s\n' \
    "$HOSTFILE" "$node_name" "$NODE_RANK" "$NUM_NODES" "$HEAD_NODE" >&2
[[ "$MASTER_PORT" =~ ^[0-9]{1,5}$ ]] || die 'MASTER_PORT must be an integer.'
MASTER_PORT=$((10#$MASTER_PORT))
(( MASTER_PORT >= 1 && MASTER_PORT <= 65535 )) || die 'MASTER_PORT must be 1..65535.'

[[ -r "$NETWORK_SCRIPT" ]] || die "Network helper missing: $NETWORK_SCRIPT"
[[ -r "$PROJECT/.venv/bin/activate" ]] || die "Virtual environment missing: $PROJECT/.venv"
[[ -x "$PROJECT/.venv/bin/vllm" && -x "$PROJECT/.venv/bin/python" ]] || die 'vLLM/Python executable missing from the virtual environment.'
[[ -d "$MODEL_PATH" ]] || die "Local model directory missing: $MODEL_PATH"
source "$PROJECT/.venv/bin/activate"
cd -- "$PROJECT"

# Count GPUs visible to this Python process, respecting CUDA_VISIBLE_DEVICES.
NUM_GPUS=$("$PROJECT/.venv/bin/python" -c 'import torch; print(torch.cuda.device_count())')
[[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]] || die 'No visible CUDA GPUs.'
TP_SIZE=${TP_SIZE:-$NUM_GPUS}
PP_SIZE=${PP_SIZE:-$NUM_NODES}
[[ "$TP_SIZE" =~ ^[0-9]{1,6}$ && "$PP_SIZE" =~ ^[0-9]{1,6}$ ]] || die 'TP_SIZE and PP_SIZE must be positive integers.'
TP_SIZE=$((10#$TP_SIZE)); PP_SIZE=$((10#$PP_SIZE))
(( TP_SIZE > 0 && PP_SIZE > 0 && TP_SIZE * PP_SIZE == NUM_NODES * NUM_GPUS )) ||
    die 'TP_SIZE * PP_SIZE must equal NUM_NODES * visible GPUs/node for this homogeneous launcher.'

network_args=(--transport "$NET_TRANSPORT")
[[ "${NET_DEBUG:-1}" != 1 ]] || network_args+=(--debug)
[[ -z "${NET_IFACE:-}" ]] || network_args+=(--iface "$NET_IFACE")
[[ -z "${NET_HCA:-}" ]] || network_args+=(--hca "$NET_HCA")
[[ -z "${NET_LOCAL_IP:-}" ]] || network_args+=(--local-ip "$NET_LOCAL_IP")
[[ "${NET_USE_MASTER_ROUTE:-0}" != 1 ]] || network_args+=(--master "$HEAD_IP")
if (( NODE_RANK == 0 )); then
    network_args+=(--http --http-reserved-ports "$MASTER_PORT${HTTP_RESERVED_PORTS:+,$HTTP_RESERVED_PORTS}")
    if [[ -n "${SERVICE_PORT:-}" ]]; then
        network_args+=(--http-port "$SERVICE_PORT")
    else
        network_args+=(--http-ports "$HTTP_PORTS")
    fi
    [[ -z "${HTTP_IFACE:-}" ]] || network_args+=(--http-iface "$HTTP_IFACE")
    [[ -z "${HTTP_CLIENT:-}" ]] || network_args+=(--http-client "$HTTP_CLIENT")
    [[ -z "${HTTP_IP:-}" ]] || network_args+=(--http-ip "$HTTP_IP")
    [[ "${HTTP_AUDIT:-0}" != 1 ]] || network_args+=(--http-audit)
fi
source "$NETWORK_SCRIPT" "${network_args[@]}" || die 'Network configuration failed; vLLM was not launched.'

if (( NODE_RANK == 0 )); then
    # Ensure HEAD_IP is assigned locally and catch stale listeners before loading
    # weights. A wildcard bind also detects conflicts on another local address.
    "$PROJECT/.venv/bin/python" - "$HEAD_IP" "$MASTER_PORT" <<'PY'
import ipaddress, socket, sys
try:
    address = ipaddress.IPv4Address(sys.argv[1])
    if address.is_unspecified or address.is_loopback or address.is_multicast:
        raise ValueError('HEAD_IP must be a peer-reachable unicast IPv4 address')
    for host in (str(address), '0.0.0.0'):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, int(sys.argv[2])))
except (OSError, ValueError) as exc:
    sys.exit(f'Rendezvous preflight failed: {exc}')
PY
fi

vllm_args=(serve "$MODEL_PATH"
    --tensor-parallel-size "$TP_SIZE" --pipeline-parallel-size "$PP_SIZE"
    --distributed-executor-backend mp --nnodes "$NUM_NODES" --node-rank "$NODE_RANK"
    --master-addr "$HEAD_IP" --master-port "$MASTER_PORT"
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTILIZATION"
    --max-num-seqs "$MAX_NUM_SEQS" --load-format "$LOAD_FORMAT" --enable-prefix-caching)
[[ -z "${MOE_BACKEND:-}" ]] || vllm_args+=(--moe-backend "$MOE_BACKEND")
[[ "${ENFORCE_EAGER:-0}" != 1 ]] || vllm_args+=(--enforce-eager)
if (( NODE_RANK == 0 )); then
    [[ "$INFER_HTTP_PORT" != "$MASTER_PORT" ]] || die 'HTTP and rendezvous ports must differ.'
    vllm_args+=(--served-model-name "$MODEL_NAME"
        --host "$INFER_HTTP_HOST" --port "$INFER_HTTP_PORT"
        --enable-auto-tool-choice --tool-call-parser qwen3_coder)
    printf 'HTTP candidate endpoint: %s (external reachability unverified)\n' "$INFER_HTTP_URL" >&2
else
    vllm_args+=(--headless)
fi
printf 'Node %s: rank=%s/%s, TP=%s PP=%s, rendezvous=%s:%s\n' \
    "$node_name" "$NODE_RANK" "$NUM_NODES" "$TP_SIZE" "$PP_SIZE" "$HEAD_IP" "$MASTER_PORT" >&2
printf 'Command: ' >&2
printf '%q ' "$PROJECT/.venv/bin/vllm" "${vllm_args[@]}" >&2
printf '\n' >&2
(( ! dry_run )) || exit 0
exec "$PROJECT/.venv/bin/vllm" "${vllm_args[@]}"

