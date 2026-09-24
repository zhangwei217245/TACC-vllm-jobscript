#!/usr/bin/env bash
# Per-node Qwen3-Coder-Next launcher. Bash >=4; NVIDIA CUDA; local vLLM venv.
# Run on EVERY node. This script launches no remote processes and requests no allocation.
# Keep the repository layout and the SAME ordered hosts.txt on all nodes.
# The network helper lives at ../utility/inference-network.sh relative to this script.
# The helper configures local NICs; the first host in hosts.txt is the head.
# All nodes need identical model weights/configuration and serving settings.
#
# Defaults retained: local model name, HEAD_IP=192.168.1.1,
# MASTER_PORT=8041, HTTP range 8040-8050, RDMA, instanttensor, memory fraction .75.
# Defaults: TP=2, PP=2 (four GPU ranks); 1M context uses YaRN; max sequences=1.
# Four one-GPU nodes: TP spans pairs of nodes; PP has two stages.
# 1M is experimental extension of the native 256K window, not a quality guarantee.
# Use CONTEXT_PROFILE=128k or 256k for ordinary coding without RoPE scaling.
#
# Examples (use the same overrides on every node):
#   bash launch-vllm.sh --help
#   bash launch-vllm.sh --dry-run
#   CONTEXT_PROFILE=128k bash launch-vllm.sh
#   CONTEXT_PROFILE=512k bash launch-vllm.sh
#   CONTEXT_PROFILE=1m KV_CACHE_DTYPE=fp8 bash launch-vllm.sh
#   NET_TRANSPORT=socket TP_SIZE=4 PP_SIZE=1 bash launch-vllm.sh
#   TP_SIZE=4 PP_SIZE=1 CONTEXT_PROFILE=128k SPEC_METHOD=ngram bash launch-vllm.sh
# Download a complete draft checkpoint once into shared storage (or on each node):
#   hf download z-lab/Qwen3-Coder-Next-DFlash \
#     --local-dir "$PROJECT/models/Qwen3-Coder-Next-DFlash"
# Downloading locally does not establish FP8/GB10 compatibility.
# Use the same absolute directory on every node:
#   TP_SIZE=4 PP_SIZE=1 CONTEXT_PROFILE=128k SPEC_METHOD=dflash \
#     SPEC_MODEL="$PROJECT/models/Qwen3-Coder-Next-DFlash" \
#     bash launch-vllm.sh
#   export VLLM_API_KEY='private-key'  # optional; never printed by this launcher
# Static UI on the head's existing HTTP port (enabled by default for this kit):
#   VLLM_UI_ENABLE=0 bash dgxspark/launch-vllm.sh  # Disable UI.
#   VLLM_UI_PAGE=chat.html bash dgxspark/launch-vllm.sh
#   VLLM_PUBLIC_BASE_URL=https://llm.example.org bash dgxspark/launch-vllm.sh
# Layout: dgxspark/launch-vllm.sh, vllm_middleware/static_ui.py, ui/chat.html.
# The page reads public API URL/model settings from /ui/config.json; no API key.
# Without VLLM_PUBLIC_BASE_URL, the UI uses the browser's current server address.
# Generate hosts.txt ONCE per Slurm allocation; distribute the same ordering:
#   scontrol show hostnames "$SLURM_JOB_NODELIST" > hosts.txt
#
# Reviewed 2026-09-24 against current upstream source; no GPU validation.
# PP + model-based speculation also requires model-specific auxiliary-state relay.
# Qwen3NextModel currently lacks that relay; the installed-build preflight checks it.
# https://huggingface.co/Qwen/Qwen3-Coder-Next
# https://docs.vllm.ai/en/v0.29.0/serving/parallelism_scaling/
# https://docs.vllm.ai/en/latest/features/context_extension/
# https://docs.vllm.ai/en/latest/features/speculative_decoding/
# https://huggingface.co/z-lab/Qwen3-Coder-Next-DFlash
# https://huggingface.co/togethercomputer/Aurora-Spec-Qwen3-Coder-Next-FP8

set -euo pipefail
(( BASH_VERSINFO[0] >= 4 )) || { printf 'Bash 4 or later is required.\n' >&2; exit 1; }
die() { printf 'launch-vllm: %s\n' "$*" >&2; exit 1; }
warn() { printf 'launch-vllm: %s\n' "$*" >&2; }
positive_int() {
    local name=$1 value=${!1}
    [[ $value =~ ^[0-9]{1,9}$ ]] || die "$name must be a positive integer."
    (( 10#$value > 0 )) || die "$name must be positive."
    printf -v "$name" '%d' "$((10#$value))"
}
boolean() { [[ ${!1} == 0 || ${!1} == 1 ]] || die "$1 must be 0 or 1."; }

dry_run=0
case ${1:-} in
    --dry-run) dry_run=1; shift ;;
    -h|--help)
        cat <<'HELP'
Usage: bash launch-vllm.sh [--dry-run]
Run once on EACH node. No SSH, Ray cluster creation, or Slurm allocation is done.
Dry-run runs local checks (including the network helper) and prints a redacted
command. It loads no weights, starts no vLLM, and does not reserve ports.

Paths and membership:
  PROJECT=$DEPLOY_KIT_ROOT            (repo root; .venv/bin/python and vllm required)
  NETWORK_SCRIPT=$DEPLOY_KIT_ROOT/utility/inference-network.sh
  HOSTFILE=hosts.txt                  (beside launcher unless overridden)
  LOCAL_NODE_NAME                     (optional override for Slurm aliases)
  HEAD_IP=192.168.1.1 MASTER_PORT=8041 (same on all nodes; head owns HEAD_IP)
  MODEL_NAME=Qwen--Qwen3-Coder-Next-FP8 MODEL_REPO=$PROJECT/models MODEL_PATH=...
  SERVED_MODEL_NAME                   (default MODEL_NAME without author-- prefix)

Context and performance:
  CONTEXT_PROFILE=128k|256k|512k|1m    (default 1m; explicit MAX_MODEL_LEN wins)
  MAX_MODEL_LEN                      (total input + output tokens)
  VLLM_ALLOW_LONG_MAX_MODEL_LEN        (defaults to 1 above native context, with YaRN)
  MAX_NUM_SEQS                        (default 1 above native context, otherwise 4)
  MAX_NUM_BATCHED_TOKENS              (default 4096 extended, otherwise 8192)
  BATCH_TOKENS                        (fallback alias for the setting above)
  GPU_MEM_UTILIZATION=0.75 DTYPE=auto KV_CACHE_DTYPE=auto LOAD_FORMAT=instanttensor
  TP_SIZE=2 PP_SIZE=2                 (four GPU ranks; environment overrides allowed)
  TP_SIZE * PP_SIZE must equal NUM_NODES * visible GPUs per node.
  PREFIX_CACHING=1 CHUNKED_PREFILL=1 ENFORCE_EAGER=0
  ATTENTION_BACKEND MOE_BACKEND MAMBA_CACHE_MODE (optional; retain auto selection)
  HF_OVERRIDES                       (optional JSON object, merged over auto YaRN)

Tools, sampling, and API:
  TOOL_CALL_PARSER=qwen3_coder ENABLE_AUTO_TOOL_CHOICE=1 PROMPT_TOKENS_DETAILS=1
  TEMPERATURE=1.0 TOP_P=0.95 TOP_K=40 REPETITION_PENALTY=1.0
  PRESENCE_PENALTY=0.0 FREQUENCY_PENALTY=0.0
  VLLM_API_KEY                        (optional, passed only on head; redacted)
  No reasoning parser: official Coder-Next is a non-thinking checkpoint.

Static web UI (optional, head only):
  DEPLOY_KIT_ROOT                     (default parent of this script's directory)
  VLLM_UI_ENABLE=1                    (0 disables the UI)
  VLLM_MIDDLEWARE_DIR=$DEPLOY_KIT_ROOT/vllm_middleware
  VLLM_UI_DIR=$DEPLOY_KIT_ROOT/ui      (only this directory is served)
  VLLM_UI_PAGE=chat.html              (entry HTML file relative to VLLM_UI_DIR)
  VLLM_PUBLIC_BASE_URL                (optional browser-facing server root, WITHOUT /v1)
  Example: https://llm.example.org or https://gateway.example.org/qwen
  Empty public URL means same-origin; no dependence on bind address or local NIC.
  Relative directory overrides use caller cwd. Defaults use repo layout above.
  Serves /ui/ -> /ui/chat.html and /ui/config.json. No SPA route fallback.
  UI files are public; API requests still pass through vLLM authentication.
  Serve only frontend build output. Never put API keys in bundled UI assets.
  Workers need neither the UI directory nor middleware.

Speculative decoding (default off):
  SPEC_METHOD=none|ngram|dflash|eagle3
  SPEC_TP_SIZE=1                     (draft TP; model-based methods only)
  TACC_QWEN3NEXT_PP_DFLASH=0          (opt-in experimental v0.30.0 source patch;
    apply utility/qwen3next_pp_patch.py first on every node; see patches/qwen3next-pp)
  SPEC_TOKENS                        (defaults: ngram=4, dflash=15, eagle3=3)
  SPEC_MODEL                         (Hugging Face ID or absolute local draft directory)
  NGRAM_MIN=2 NGRAM_MAX=5
  DFlash: z-lab/Qwen3-Coder-Next-DFlash (author recipe: BF16 target, FlashAttention,
    batch tokens 32768; FP8/GB10 combinations require validation).
  EAGLE3: togethercomputer/Aurora-Spec-Qwen3-Coder-Next-FP8 (experimental in vLLM;
    checkpoint author documents SGLang). Drafter context may be shorter than target.
  MTP/DSpark are not presets: no matching native MTP/DSpark checkpoint verified.
  DFlash/EAGLE3 with PP>1: requires Model Runner V2, draft PP=1, and Qwen3Next
  auxiliary-state relay support. Checked in the installed build before loading.
  Stock v0.30.0 Qwen3Next lacks that relay. The opt-in prototype requires
  TP=1, PP=2/4, draft TP=1, DFlash, native context, and eager execution.
  Otherwise use SPEC_METHOD=none for TP=2/PP=2, or TP=4/PP=1 for drafting.
  N-gram is not implemented in the checked V2 runner; this preset uses PP=1.

Existing network helper options (forwarded unchanged):
  NET_TRANSPORT=rdma|auto|socket NET_DEBUG=1 NET_IFACE NET_HCA NET_LOCAL_IP
  NET_USE_MASTER_ROUTE=0              (1 passes HEAD_IP to helper's --master)
  HTTP_PORTS=8040-8050 SERVICE_PORT HTTP_IFACE HTTP_CLIENT HTTP_IP HTTP_AUDIT=0
  HTTP_RESERVED_PORTS                 (MASTER_PORT always excluded)
  Helper must EXPORT/set INFER_HTTP_HOST and INFER_HTTP_PORT on rank 0.

hosts.txt accepts one expanded hostname per line; blank/comment lines ignored.
Short labels must be unique (case-insensitive). NODE_RANK/NUM_NODES are derived;
conflicting environment values fail. GPU count must be homogeneous across nodes.
Compare the printed configuration fingerprints manually; no remote consistency
check is performed. A free bind port does not prove external client reachability.
Keep the endpoint private or behind a gateway: --api-key is not whole-server TLS
or authorization. KV offloading is deliberately unset, including at 1M.
HELP
    exit 0 ;;
esac
(( $# == 0 )) || die 'Unknown argument; use --help. Configure through environment variables.'

# Resolve caller-relative paths before moving into the project directory.
LAUNCH_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DEPLOY_KIT_ROOT=${DEPLOY_KIT_ROOT:-$(cd -- "$LAUNCH_DIR/.." && pwd)}
[[ $DEPLOY_KIT_ROOT == /* ]] || DEPLOY_KIT_ROOT=$PWD/$DEPLOY_KIT_ROOT
VLLM_UI_ENABLE=${VLLM_UI_ENABLE:-1}
boolean VLLM_UI_ENABLE
VLLM_MIDDLEWARE_DIR=${VLLM_MIDDLEWARE_DIR:-$DEPLOY_KIT_ROOT/vllm_middleware}
VLLM_UI_DIR=${VLLM_UI_DIR:-$DEPLOY_KIT_ROOT/ui}
VLLM_UI_PAGE=${VLLM_UI_PAGE:-chat.html}
VLLM_PUBLIC_BASE_URL=${VLLM_PUBLIC_BASE_URL:-}
for path_name in VLLM_MIDDLEWARE_DIR VLLM_UI_DIR; do
    [[ ${!path_name} == /* ]] || printf -v "$path_name" '%s/%s' "$PWD" "${!path_name}"
done
PROJECT=${PROJECT:-$DEPLOY_KIT_ROOT}
[[ -d $PROJECT ]] || die "Project directory missing: $PROJECT"
PROJECT=$(cd -- "$PROJECT" && pwd)
TACC_QWEN3NEXT_PP_DFLASH=${TACC_QWEN3NEXT_PP_DFLASH:-0}
boolean TACC_QWEN3NEXT_PP_DFLASH
export TACC_QWEN3NEXT_PP_DFLASH
QWEN3NEXT_PATCH_ID=disabled
NETWORK_SCRIPT=${NETWORK_SCRIPT:-$DEPLOY_KIT_ROOT/utility/inference-network.sh}
HOSTFILE=${HOSTFILE:-$LAUNCH_DIR/hosts.txt}
MODEL_NAME=${MODEL_NAME:-Qwen--Qwen3-Coder-Next-FP8}
MODEL_REPO=${MODEL_REPO:-$PROJECT/models}
MODEL_PATH=${MODEL_PATH:-$MODEL_REPO/$MODEL_NAME}
for path_name in NETWORK_SCRIPT HOSTFILE MODEL_PATH; do
    [[ ${!path_name} == /* ]] || printf -v "$path_name" '%s/%s' "$PWD" "${!path_name}"
done
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-${MODEL_NAME#*--}}
HEAD_IP=${HEAD_IP:-192.168.1.1}
MASTER_PORT=${MASTER_PORT:-8041}
positive_int MASTER_PORT
(( MASTER_PORT <= 65535 )) || die 'MASTER_PORT must be 1..65535.'
[[ -r $HOSTFILE ]] || die "Host file missing or unreadable: $HOSTFILE"

# Derive rank from file ordering, never from whichever NIC the helper selects.
node_name=${LOCAL_NODE_NAME:-$(hostname -s)}
local_key=${node_name,,}; local_key=${local_key%%.*}
[[ -n $local_key ]] || die 'Cannot determine the local hostname.'
declare -a NODES=()
declare -A seen_hosts=()
detected_rank=-1
line_number=0
while IFS= read -r host_line || [[ -n $host_line ]]; do
    line_number=$((line_number + 1))
    host_line=${host_line%$'\r'}
    host_entry=; extra=
    read -r host_entry extra <<< "$host_line" || true
    [[ -n $host_entry && $host_entry != \#* ]] || continue
    [[ -z $extra && $host_entry =~ ^[[:alnum:]_][[:alnum:]_.-]*$ ]] ||
    die "Invalid host at $HOSTFILE:$line_number; one expanded hostname per line required."
    host_key=${host_entry,,}; host_key=${host_key%%.*}
    [[ -z ${seen_hosts[$host_key]+present} ]] || die "Duplicate short hostname '$host_key'."
    seen_hosts[$host_key]=1
    [[ $host_key != "$local_key" ]] || detected_rank=${#NODES[@]}
    NODES+=("$host_entry")
done < "$HOSTFILE"
(( ${#NODES[@]} > 0 )) || die 'Host file is empty.'
(( detected_rank >= 0 )) || die "Local host '$node_name' absent from host file; check LOCAL_NODE_NAME."
for derived_name in NODE_RANK NUM_NODES; do
    supplied_value=${!derived_name-}
    expected_value=$detected_rank
    [[ $derived_name != NUM_NODES ]] || expected_value=${#NODES[@]}
    if [[ -n $supplied_value ]]; then
        [[ $supplied_value =~ ^[0-9]{1,6}$ ]] || die "$derived_name must be an integer."
        (( 10#$supplied_value == expected_value )) ||
        die "$derived_name=$supplied_value conflicts with host file value $expected_value."
    fi
done
NODE_RANK=$detected_rank
NUM_NODES=${#NODES[@]}
HEAD_NODE=${NODES[0]}

[[ -r $NETWORK_SCRIPT ]] || die "Network helper missing: $NETWORK_SCRIPT"
[[ -r $PROJECT/.venv/bin/activate ]] || die "Virtual environment missing: $PROJECT/.venv"
PYTHON=$PROJECT/.venv/bin/python
VLLM=$PROJECT/.venv/bin/vllm
[[ -x $PYTHON && -x $VLLM ]] || die 'vLLM/Python executable missing from virtual environment.'
[[ -r $MODEL_PATH/config.json ]] || die "Model config missing: $MODEL_PATH/config.json"
source "$PROJECT/.venv/bin/activate"
cd -- "$PROJECT"
# Validate optional frontend before CUDA/model startup. Workers have no HTTP app.
if (( NODE_RANK == 0 && VLLM_UI_ENABLE )); then
    [[ -r $VLLM_MIDDLEWARE_DIR/static_ui.py ]] || die "Middleware missing: $VLLM_MIDDLEWARE_DIR/static_ui.py"
    # Keep project-owned settings out of vLLM's reserved VLLM_* namespace.
    export TACC_UI_DIR=$VLLM_UI_DIR TACC_UI_PAGE=$VLLM_UI_PAGE
    export TACC_PUBLIC_BASE_URL=$VLLM_PUBLIC_BASE_URL
    export TACC_UI_MODEL=$SERVED_MODEL_NAME  # Public model ID, never the API key.
    # Import static_ui from the dedicated middleware directory; no __init__.py needed.
    export PYTHONPATH="$VLLM_MIDDLEWARE_DIR${PYTHONPATH:+:$PYTHONPATH}"
    "$PYTHON" - "$VLLM_MIDDLEWARE_DIR/static_ui.py" <<'PY'
import pathlib, sys
import static_ui
if pathlib.Path(static_ui.__file__).resolve() != pathlib.Path(sys.argv[1]).resolve():
    sys.exit('Another static_ui module shadows the launcher middleware; check PROJECT/PYTHONPATH.')
static_ui.StaticUIMiddleware(app=None)  # Checks dependencies and static directory.
PY
fi
# Retain legacy launcher inputs as shell variables, but do not export them to vLLM.
export -n VLLM_UI_ENABLE VLLM_UI_MODEL VLLM_UI_DIR VLLM_UI_PAGE \
VLLM_PUBLIC_BASE_URL VLLM_MIDDLEWARE_DIR
NUM_GPUS=$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')
positive_int NUM_GPUS
TP_SIZE=${TP_SIZE:-2}  # Tensor shards per pipeline stage.
PP_SIZE=${PP_SIZE:-2}  # Pipeline stages; TP=2/PP=2 needs four GPU ranks.
positive_int TP_SIZE
positive_int PP_SIZE
(( TP_SIZE * PP_SIZE == NUM_NODES * NUM_GPUS )) ||
die "TP_SIZE=$TP_SIZE * PP_SIZE=$PP_SIZE must equal NUM_NODES=$NUM_NODES * visible GPUs/node=$NUM_GPUS."

# Explicit MAX_MODEL_LEN overrides the preset. Auto YaRN is based on this final length.
CONTEXT_PROFILE=${CONTEXT_PROFILE:-1m}
case ${CONTEXT_PROFILE,,} in
    128k) profile_len=131072 ;;
    256k) profile_len=262144 ;;
    512k) profile_len=524288 ;;
    1m) profile_len=1048576 ;;
    *) die 'CONTEXT_PROFILE must be 128k, 256k, 512k, or 1m.' ;;
esac
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$profile_len}
positive_int MAX_MODEL_LEN
GPU_MEM_UTILIZATION=${GPU_MEM_UTILIZATION:-0.75}
DTYPE=${DTYPE:-auto}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
LOAD_FORMAT=${LOAD_FORMAT:-instanttensor}
PREFIX_CACHING=${PREFIX_CACHING:-1}
CHUNKED_PREFILL=${CHUNKED_PREFILL:-1}
ENFORCE_EAGER=${ENFORCE_EAGER:-$TACC_QWEN3NEXT_PP_DFLASH}
ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE:-1}
PROMPT_TOKENS_DETAILS=${PROMPT_TOKENS_DETAILS:-1}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-qwen3_coder}
NET_TRANSPORT=${NET_TRANSPORT:-rdma}
NET_DEBUG=${NET_DEBUG:-1}
NET_USE_MASTER_ROUTE=${NET_USE_MASTER_ROUTE:-0}
HTTP_AUDIT=${HTTP_AUDIT:-0}
HTTP_PORTS=${HTTP_PORTS:-8040-8050}
for name in PREFIX_CACHING CHUNKED_PREFILL ENFORCE_EAGER ENABLE_AUTO_TOOL_CHOICE \
PROMPT_TOKENS_DETAILS NET_DEBUG NET_USE_MASTER_ROUTE HTTP_AUDIT; do boolean "$name"; done
case $NET_TRANSPORT in rdma|auto|socket) ;; *) die 'Invalid NET_TRANSPORT.' ;; esac

# Build JSON with Python, not shell interpolation/eval. Preserve checkpoint RoPE values.
# User HF_OVERRIDES recursively overrides auto settings; it must be a JSON object.
config_output=$("$PYTHON" - "$MODEL_PATH/config.json" "$MAX_MODEL_LEN" "${HF_OVERRIDES:-}" \
"$GPU_MEM_UTILIZATION" "${TEMPERATURE:-1.0}" "${TOP_P:-0.95}" "${TOP_K:-40}" \
    "${REPETITION_PENALTY:-1.0}" "${PRESENCE_PENALTY:-0.0}" "${FREQUENCY_PENALTY:-0.0}" <<'PY'
import hashlib, json, math, sys
try:
    path, length, overrides, mem, temp, top_p, top_k, rep, pres, freq = sys.argv[1:]
    with open(path) as f:
        model = json.load(f)
    native = int(model['max_position_embeddings'])
    length = int(length)
    if native <= 0:
        raise ValueError('native context must be positive')
    mem, temp, top_p, rep, pres, freq = map(float, (mem, temp, top_p, rep, pres, freq))
    top_k = int(top_k)
    if not all(math.isfinite(v) for v in (mem, temp, top_p, rep, pres, freq)):
        raise ValueError('numeric settings must be finite')
    if not (0 < mem <= 1 and temp >= 0 and 0 < top_p <= 1 and rep > 0):
        raise ValueError('invalid memory utilization or sampling range')
    if top_k != -1 and top_k < 1:
        raise ValueError('TOP_K must be -1 or positive')
    if not (-2 <= pres <= 2 and -2 <= freq <= 2):
        raise ValueError('presence/frequency penalties must be within [-2, 2]')
    user = json.loads(overrides) if overrides else {}
    if not isinstance(user, dict):
        raise ValueError('HF_OVERRIDES must be a JSON object')
    hf = {}
    if length > native:
        if model.get('model_type') != 'qwen3_next':
            raise ValueError('automatic YaRN is specific to qwen3_next; use an appropriate model launcher')
        rope = model.get('rope_scaling') or model.get('rope_parameters') or {}
        if rope.get('rope_type', rope.get('type', 'default')) != 'default':
            raise ValueError('checkpoint already scales RoPE; avoid automatically scaling it a second time')
        hf = {'rope_parameters': {
            'rope_type': 'yarn', 'factor': length / native,
            'original_max_position_embeddings': native,
            'rope_theta': rope.get('rope_theta', model.get('rope_theta', 5000000.0)),
            'partial_rotary_factor': rope.get('partial_rotary_factor', model.get('partial_rotary_factor', 0.25)),
        }}
    def merge(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v
    merge(hf, user)
    sampling = dict(temperature=temp, top_p=top_p, top_k=top_k,
                    repetition_penalty=rep, presence_penalty=pres, frequency_penalty=freq)
    print(native)
    print(json.dumps(hf, separators=(',', ':'), allow_nan=False))
    print(json.dumps(sampling, separators=(',', ':'), allow_nan=False))
    print(hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest())
except (ValueError, KeyError, TypeError, OSError) as exc:
    sys.exit(f'Configuration error: {exc}')
PY
) || die 'Model/context/sampling validation failed.'
mapfile -t config_lines <<< "$config_output"
NATIVE_CONTEXT=${config_lines[0]}
HF_CONFIG=${config_lines[1]}
GENERATION_CONFIG=${config_lines[2]}
MODEL_CONFIG_SHA=${config_lines[3]}
if (( MAX_MODEL_LEN > NATIVE_CONTEXT )); then
    # Permit the extended limit; HF_CONFIG supplies the actual YaRN scaling.
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}
    boolean VLLM_ALLOW_LONG_MAX_MODEL_LEN
    default_seqs=1; default_batch=4096
    warn "Extended context $MAX_MODEL_LEN > native $NATIVE_CONTEXT: validate quality and memory."
else
    default_seqs=4; default_batch=8192
fi
MAX_NUM_SEQS=${MAX_NUM_SEQS:-$default_seqs}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-${BATCH_TOKENS:-$default_batch}}
positive_int MAX_NUM_SEQS
positive_int MAX_NUM_BATCHED_TOKENS
(( MAX_NUM_BATCHED_TOKENS >= MAX_NUM_SEQS )) || die 'Batch-token budget must cover MAX_NUM_SEQS.'
if (( ! CHUNKED_PREFILL && MAX_NUM_BATCHED_TOKENS < MAX_MODEL_LEN )); then
    die 'Without chunked prefill, batch-token budget must cover MAX_MODEL_LEN.'
fi

# Exactly one speculative method. Native MTP/DSpark support is not assumed.
SPEC_METHOD=${SPEC_METHOD:-none}
if (( TACC_QWEN3NEXT_PP_DFLASH )); then
    [[ $SPEC_METHOD == dflash ]] || die 'Experimental relay requires SPEC_METHOD=dflash.'
    (( TP_SIZE == 1 && (PP_SIZE == 2 || PP_SIZE == 4) )) ||
        die 'Experimental relay requires TP_SIZE=1 and PP_SIZE=2 or 4.'
    (( ENFORCE_EAGER == 1 )) || die 'Experimental relay currently requires ENFORCE_EAGER=1.'
    (( MAX_MODEL_LEN <= NATIVE_CONTEXT )) && [[ $HF_CONFIG == '{}' ]] ||
        die 'Experimental relay requires native context and no HF_OVERRIDES; start with CONTEXT_PROFILE=128k.'
    "$PYTHON" "$DEPLOY_KIT_ROOT/utility/qwen3next_pp_patch.py" --check >&2 ||
        die 'Install the pinned relay patch in this serving environment before enabling it.'
    QWEN3NEXT_PATCH_ID=qwen3next-dflash-pp-v030-r1
    warn "EXPERIMENTAL $QWEN3NEXT_PATCH_ID: GPU correctness and throughput remain unverified."
fi
SPEC_CONFIG=
case $SPEC_METHOD in
    none) [[ -z ${SPEC_MODEL:-} && -z ${SPEC_TOKENS:-} ]] || die 'SPEC_MODEL/TOKENS require a speculative method.' ;;
    ngram|dflash|eagle3)
        case $SPEC_METHOD in
            ngram)
                (( PP_SIZE == 1 )) || die 'N-gram preset requires PP_SIZE=1; PP/V2 support is not established.'
                [[ ${VLLM_USE_V2_MODEL_RUNNER:-0} != 1 ]] || die 'N-gram is not supported by the checked Model Runner V2.'
                [[ -z ${SPEC_MODEL:-} ]] || die 'N-gram drafting does not use SPEC_MODEL.'
                SPEC_TOKENS=${SPEC_TOKENS:-4}; SPEC_MODEL= ;;
            dflash)
                SPEC_TOKENS=${SPEC_TOKENS:-15}
                SPEC_MODEL=${SPEC_MODEL:-z-lab/Qwen3-Coder-Next-DFlash}
                warn 'DFlash preset is experimental on FP8/GB10; verify drafter/backend compatibility.' ;;
            eagle3)
                SPEC_TOKENS=${SPEC_TOKENS:-3}
                SPEC_MODEL=${SPEC_MODEL:-togethercomputer/Aurora-Spec-Qwen3-Coder-Next-FP8}
                warn 'Aurora EAGLE3 author documents SGLang; validate this vLLM combination.' ;;
        esac
        SPEC_TP_SIZE=${SPEC_TP_SIZE:-1}
        positive_int SPEC_TP_SIZE
        if [[ $SPEC_METHOD != ngram ]]; then
            (( SPEC_TP_SIZE == 1 || SPEC_TP_SIZE == TP_SIZE )) ||
                die 'SPEC_TP_SIZE must be 1 or match target TP_SIZE.'
            case $SPEC_MODEL in
                /*) [[ -r $SPEC_MODEL/config.json ]] || die "Draft config missing: $SPEC_MODEL/config.json" ;;
                ./*|../*|\~*) die 'Use an absolute local SPEC_MODEL path, or a Hugging Face repository ID.' ;;
            esac
            if (( PP_SIZE > 1 )); then
                export VLLM_USE_V2_MODEL_RUNNER=${VLLM_USE_V2_MODEL_RUNNER:-1}
                [[ $VLLM_USE_V2_MODEL_RUNNER == 1 ]] ||
                    die 'Model-based speculation with PP>1 requires VLLM_USE_V2_MODEL_RUNNER=1.'
            fi
        fi
        positive_int SPEC_TOKENS
        NGRAM_MIN=${NGRAM_MIN:-2}; NGRAM_MAX=${NGRAM_MAX:-5}
        positive_int NGRAM_MIN; positive_int NGRAM_MAX
        (( NGRAM_MIN <= NGRAM_MAX )) || die 'NGRAM_MIN must not exceed NGRAM_MAX.'
        SPEC_CONFIG=$("$PYTHON" - "$SPEC_METHOD" "$SPEC_TOKENS" "$SPEC_MODEL" "$NGRAM_MIN" "$NGRAM_MAX" "$SPEC_TP_SIZE" "$TP_SIZE" "$PP_SIZE" "$NUM_NODES" "$MODEL_PATH" <<'PY'
import contextlib, json, sys
method, tokens, model, low, high, draft_tp, target_tp, target_pp, nodes, target_path = sys.argv[1:]
config = dict(method=method, num_speculative_tokens=int(tokens))
if method == 'ngram':
    config.update(prompt_lookup_min=int(low), prompt_lookup_max=int(high))
else:
    config['model'] = model
    config['draft_tensor_parallel_size'] = int(draft_tp)
    if int(target_pp) > 1:
        # No weights instantiated. Send import/config logs to stderr so the shell
        # captures exactly one JSON document, even if vLLM logs to stdout.
        try:
            with contextlib.redirect_stdout(sys.stderr):
                from vllm.config import ParallelConfig, SpeculativeConfig
                from vllm.v1.worker.gpu import model_runner
                from vllm.model_executor.models.qwen3_next import Qwen3NextModel
                import os
                if os.environ.get('TACC_QWEN3NEXT_PP_DFLASH') == '1':
                    if getattr(Qwen3NextModel, '_tacc_pp_patch', None) != 'qwen3next-dflash-pp-v030-r1':
                        raise RuntimeError('experimental patch was not loaded by this Python process')
                    with open(os.path.join(target_path, 'config.json')) as f:
                        if json.load(f).get('model_type') != 'qwen3_next':
                            raise RuntimeError('experimental relay only supports model_type=qwen3_next')
                if not getattr(Qwen3NextModel, 'supports_aux_hidden_states_over_pp', False):
                    raise RuntimeError('installed Qwen3NextModel lacks auxiliary hidden-state relay across PP stages')
                target = ParallelConfig(tensor_parallel_size=int(target_tp),
                                        pipeline_parallel_size=int(target_pp),
                                        nnodes=int(nodes), distributed_executor_backend='mp')
                draft = SpeculativeConfig.create_draft_parallel_config(target, int(draft_tp))
                if draft.pipeline_parallel_size != 1:
                    raise RuntimeError('this build inherits target PP for the draft')
                if not callable(getattr(model_runner, 'verify_supports_aux_hidden_states_over_pp', None)):
                    raise RuntimeError('Model Runner V2 lacks auxiliary hidden-state relay validation')
        except (ImportError, AttributeError, TypeError, ValueError, RuntimeError) as exc:
            sys.exit(f'PP/speculation preflight failed: {exc}. Use SPEC_METHOD=none, '
                     'or PP_SIZE=1 and TP_SIZE equal to the total GPU count. '
                     'PP drafting requires a build implementing model-specific relay; '
                     'setting a capability flag alone does not implement it.')
        print(f'Speculation capability preflight passed: target TP={target_tp} PP={target_pp}; '
              f'draft TP={draft_tp} PP=1. Runtime/weights/backend compatibility still '
              'requires an actual inference test.', file=sys.stderr)
print(json.dumps(config, separators=(',', ':')))
PY
        ) || die 'Could not build speculative configuration.'
        if [[ $SPEC_METHOD != ngram ]] && (( MAX_MODEL_LEN > NATIVE_CONTEXT )); then
            warn 'Target YaRN does not extend the drafter; establish a non-speculative long-context baseline first.'
        fi ;;
    *) die 'SPEC_METHOD must be none, ngram, dflash, or eagle3.' ;;
esac

# Same engine arguments on all ranks. Frontend-only options are added on rank 0.
engine_args=(
    --tensor-parallel-size "$TP_SIZE"             # Total ranks sharding tensors.
    --pipeline-parallel-size "$PP_SIZE"           # Pipeline stages (default 2).
    --distributed-executor-backend mp             # Per-node multiprocessing, no Ray.
    --nnodes "$NUM_NODES"                        # Number of participating hosts.
    --master-addr "$HEAD_IP"                      # Shared rendezvous address, owned by head.
    --master-port "$MASTER_PORT"                  # Shared rendezvous TCP port, not HTTP.
    --dtype "$DTYPE"                             # Compute/unquantized dtype; retains FP8 weights.
    --max-model-len "$MAX_MODEL_LEN"              # Total input plus output tokens per request.
    --gpu-memory-utilization "$GPU_MEM_UTILIZATION" # Per-GPU executor memory budget fraction.
    --max-num-seqs "$MAX_NUM_SEQS"                # Maximum scheduled sequences; others may queue.
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" # Per-step token budget, not context size.
    --kv-cache-dtype "$KV_CACHE_DTYPE"            # Attention KV precision, separate from weights.
    --load-format "$LOAD_FORMAT"                 # Weight loader; affects startup, not context.
    --generation-config vllm                     # Use explicit sampling defaults below.
    --override-generation-config "$GENERATION_CONFIG" # Request values may override defaults.
)
if (( PREFIX_CACHING )); then
    engine_args+=(--enable-prefix-caching)        # Reuse matching prefixes between agent turns.
else
    engine_args+=(--no-enable-prefix-caching)
fi
if (( CHUNKED_PREFILL )); then
    engine_args+=(--enable-chunked-prefill)       # Split long prompt processing into chunks.
else
    engine_args+=(--no-enable-chunked-prefill)
fi
[[ $HF_CONFIG == '{}' ]] || engine_args+=(--hf-overrides "$HF_CONFIG") # APPLY YaRN on every rank.
[[ -z $SPEC_CONFIG ]] || engine_args+=(--speculative-config "$SPEC_CONFIG") # Draft/verify tokens.
[[ -z ${ATTENTION_BACKEND:-} ]] || engine_args+=(--attention-backend "$ATTENTION_BACKEND") # Attention kernel.
[[ -z ${MOE_BACKEND:-} ]] || engine_args+=(--moe-backend "$MOE_BACKEND") # MoE kernel selection.
[[ -z ${MAMBA_CACHE_MODE:-} ]] || engine_args+=(--mamba-cache-mode "$MAMBA_CACHE_MODE") # Recurrent cache policy.
(( ! ENFORCE_EAGER )) || engine_args+=(--enforce-eager) # Debug fallback: disable graph execution.

# Fingerprint excludes node rank, local model path, NIC, HTTP bind, and API key.
# It includes config.json contents, NOT the full weight files or runtime versions.
# Include runner/length environment switches; unset is distinct from explicitly 0.
fingerprint=$("$PYTHON" - "$MODEL_CONFIG_SHA" "$TOOL_CALL_PARSER" "$SERVED_MODEL_NAME" \
    "$ENABLE_AUTO_TOOL_CHOICE" "$NET_TRANSPORT" \
    "runner_v2=${VLLM_USE_V2_MODEL_RUNNER-<unset>}" \
    "qwen3next_patch=$QWEN3NEXT_PATCH_ID" \
    "allow_long=${VLLM_ALLOW_LONG_MAX_MODEL_LEN-<unset>}" \
    "${NODES[@]}" -- "${engine_args[@]}" <<'PY'
import hashlib, json, sys
print(hashlib.sha256(json.dumps(sys.argv[1:], separators=(',', ':')).encode()).hexdigest()[:16])
PY
)
printf 'Host file: %s; local=%s; head=%s; rank=%s/%s; GPUs/node=%s; TP=%s PP=%s\n' \
    "$HOSTFILE" "$node_name" "$HEAD_NODE" "$NODE_RANK" "$NUM_NODES" "$NUM_GPUS" "$TP_SIZE" "$PP_SIZE" >&2
printf 'Context=%s native=%s; sequences=%s; batch tokens=%s; speculation=%s; config fingerprint=%s\n' \
    "$MAX_MODEL_LEN" "$NATIVE_CONTEXT" "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" "$SPEC_METHOD" "$fingerprint" >&2

# Preserve the original helper interface. Workers never request HTTP discovery.
network_args=(--transport "$NET_TRANSPORT")
(( ! NET_DEBUG )) || network_args+=(--debug)
[[ -z ${NET_IFACE:-} ]] || network_args+=(--iface "$NET_IFACE")
[[ -z ${NET_HCA:-} ]] || network_args+=(--hca "$NET_HCA")
[[ -z ${NET_LOCAL_IP:-} ]] || network_args+=(--local-ip "$NET_LOCAL_IP")
(( ! NET_USE_MASTER_ROUTE )) || network_args+=(--master "$HEAD_IP")
if (( NODE_RANK == 0 )); then
    network_args+=(--http --http-reserved-ports "$MASTER_PORT${HTTP_RESERVED_PORTS:+,$HTTP_RESERVED_PORTS}")
    if [[ -n ${SERVICE_PORT:-} ]]; then
        positive_int SERVICE_PORT
        (( SERVICE_PORT <= 65535 && SERVICE_PORT != MASTER_PORT )) || die 'Invalid/conflicting SERVICE_PORT.'
        network_args+=(--http-port "$SERVICE_PORT")
    else
        network_args+=(--http-ports "$HTTP_PORTS")
    fi
    [[ -z ${HTTP_IFACE:-} ]] || network_args+=(--http-iface "$HTTP_IFACE")
    [[ -z ${HTTP_CLIENT:-} ]] || network_args+=(--http-client "$HTTP_CLIENT")
    [[ -z ${HTTP_IP:-} ]] || network_args+=(--http-ip "$HTTP_IP")
    (( ! HTTP_AUDIT )) || network_args+=(--http-audit)
fi
source "$NETWORK_SCRIPT" "${network_args[@]}" || die 'Network helper failed; vLLM was not launched.'

# Bind checks are local snapshots, not reservations or firewall/reachability tests.
if (( NODE_RANK == 0 )); then
    [[ -n ${INFER_HTTP_HOST:-} && -n ${INFER_HTTP_PORT:-} ]] || die 'Helper did not provide HTTP host/port.'
    positive_int INFER_HTTP_PORT
    (( INFER_HTTP_PORT <= 65535 && INFER_HTTP_PORT != MASTER_PORT )) || die 'Invalid/conflicting HTTP port.'
    "$PYTHON" - "$HEAD_IP" "$MASTER_PORT" "$INFER_HTTP_HOST" "$INFER_HTTP_PORT" <<'PY'
import ipaddress, socket, sys
try:
    head = ipaddress.IPv4Address(sys.argv[1])
    if head.is_unspecified or head.is_loopback or head.is_multicast:
        raise ValueError('HEAD_IP must be a peer-reachable unicast IPv4 address owned by the head')
    bindings = [(str(head), int(sys.argv[2])), ('0.0.0.0', int(sys.argv[2])),
                (sys.argv[3], int(sys.argv[4]))]
    for host, port in bindings:
        family = socket.AF_INET6 if ':' in host else socket.AF_INET
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
except (ValueError, OSError) as exc:
    sys.exit(f'Port/address preflight failed: {exc}')
PY
fi

vllm_args=(serve "$MODEL_PATH" "${engine_args[@]}" --node-rank "$NODE_RANK") # This node's rank.
if (( NODE_RANK == 0 )); then
    vllm_args+=(
        --served-model-name "$SERVED_MODEL_NAME"   # API name; same in each coding client.
        --host "$INFER_HTTP_HOST"                 # Bind address chosen by network helper.
        --port "$INFER_HTTP_PORT"                 # HTTP API port; distinct from rendezvous.
        --tool-call-parser "$TOOL_CALL_PARSER"     # Parse Qwen Coder tool syntax into API calls.
    )
    (( ! ENABLE_AUTO_TOOL_CHOICE )) || vllm_args+=(--enable-auto-tool-choice) # Model chooses tools.
    (( ! PROMPT_TOKENS_DETAILS )) || vllm_args+=(--enable-prompt-tokens-details) # Cache usage reporting.
    [[ -z ${VLLM_API_KEY:-} ]] || vllm_args+=(--api-key "$VLLM_API_KEY") # Auth on supported API routes.
    if (( VLLM_UI_ENABLE )); then
        vllm_args+=(--middleware static_ui.StaticUIMiddleware) # Serve /ui/ on the API port.
    fi
    endpoint=${INFER_HTTP_URL:-http://$INFER_HTTP_HOST:$INFER_HTTP_PORT}
    printf 'HTTP candidate: %s (external reachability unverified); model=%s\n' "$endpoint" "$SERVED_MODEL_NAME" >&2
    if (( VLLM_UI_ENABLE )); then
        ui_base=${VLLM_PUBLIC_BASE_URL:-$endpoint}
        printf 'Static UI: %s/ui/; directory=%s; entry=%s (public files)\n' "${ui_base%/}" "$VLLM_UI_DIR" "$VLLM_UI_PAGE" >&2
        if [[ -n $VLLM_PUBLIC_BASE_URL ]]; then
            printf 'UI API base: %s/v1\n' "${VLLM_PUBLIC_BASE_URL%/}" >&2
        else
            printf 'UI API base: browser same-origin address (supports SSH tunnels and public proxies).\n' >&2
        fi
    fi
    printf 'Set client context to %s including output; update its compaction threshold too.\n' "$MAX_MODEL_LEN" >&2
else
    vllm_args+=(--headless)                       # Worker engine only, no HTTP frontend.
fi

# Do not enable shell tracing: the real command includes the optional API key.
printf 'Command: %q ' "$VLLM" >&2
redact_next=0
for arg in "${vllm_args[@]}"; do
    if (( redact_next )); then
        printf '%q ' '<redacted>' >&2; redact_next=0
    else
        printf '%q ' "$arg" >&2
        [[ $arg != --api-key ]] || redact_next=1
    fi
done
printf '\n' >&2
(( ! dry_run )) || exit 0
exec "$VLLM" "${vllm_args[@]}"

# Deliberately absent: reasoning parser, enable_thinking override, custom chat
# template, trust-remote-code, forced weight quantization, CPU/NVMe KV offloading,
# expert-parallel toggle, and hybrid-cache-manager disablement. These are not
# prerequisites for the official Coder-Next baseline. Prefix caching is not KV
# offloading. Chunked prefill does not make an oversized active cache fit.
