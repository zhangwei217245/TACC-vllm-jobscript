# Serve Qwen3-Coder-Next on DGX Spark with Slurm

The batch script requests four nodes with one GPU per node and starts one
launcher per node. vLLM uses tensor parallelism across the four GPUs; only the
first allocated node runs the HTTP API and web UI.

## Repository layout and path resolution

```text
TACC-vllm-jobscript/
├── dgxspark/
│   ├── slurm-vllm.sbatch
│   └── launch-vllm.sh
├── models/                 # downloaded checkpoints
│   └── models.txt          # ordered model IDs
├── utility/
│   ├── installer.sh
│   ├── inference-network.sh
│   └── download_model.sh
├── vllm_middleware/static_ui.py
├── .venv/                  # created by installer.sh
└── ui/chat.html
```

`slurm-vllm.sbatch` sources `utility/inference-network.sh` on the first node to
find `HEAD_IP` unless supplied. It then runs `launch-vllm.sh` on every node.
Each launcher sources the helper to export local NCCL/Gloo network settings;
the head also selects an HTTP bind address and port. Sourcing retains those
exports for the vLLM process.

Both scripts default `NETWORK_SCRIPT` to
`$DEPLOY_KIT_ROOT/utility/inference-network.sh`. The launcher derives the root
from the parent of its own directory using `BASH_SOURCE[0]`, independently of
its working directory. Slurm runs a spooled copy of the batch script, so that
script instead finds the launcher from the submission directory (repository
root or `dgxspark/`), or an explicit `DEPLOY_KIT_ROOT` or `LAUNCHER` override.

Use absolute paths for overrides. A relative `NETWORK_SCRIPT` is resolved
against the caller's directory for direct launches, or `SLURM_SUBMIT_DIR` for
batch launches. `--chdir` controls logs, not helper discovery. Both scripts check
that the helper is readable; no copy or symlink into `dgxspark/` is needed.

## 1. Prepare the environment

Run these commands in Bash, replacing the checkout path:

```bash
cd /absolute/path/to/TACC-vllm-jobscript
export DEPLOY_KIT_ROOT="$PWD"
export PROJECT="$DEPLOY_KIT_ROOT"
bash "$DEPLOY_KIT_ROOT/utility/installer.sh"
```

Run the installer on a Linux DGX node with GPU access under your site's
allocation policy. It currently defaults to Python 3.14 and vLLM 0.28.0,
overridable through `PYTHON_VERSION` and `VLLM_VERSION`. It creates
`$PROJECT/.venv` and `$PROJECT/models` and checks CUDA visibility.

Keep the checkout, environment and model files available at the same absolute
paths on every node. Install separately on each node if the environment is not
shared. The log directory must be shared because every launcher reads the same
generated `hosts.txt`. Nodes need Bash 4+, iproute2, working CUDA drivers and,
for RDMA, exposed RDMA devices and drivers.

The installer and downloader resolve the repository root from their own paths,
so they work from any working directory and always use root `.venv/` and `models/`.
The launchers default `PROJECT` to `DEPLOY_KIT_ROOT`; exporting it is optional.
To serve from another existing environment and model directory, override `PROJECT`.
This serving override does not change the installer or downloader destinations.

## 2. Download the model

Edit `models/models.txt`, with one Hugging Face `organization/model` ID per line:

```text
Qwen/Qwen3-Coder-Next-FP8
z-lab/Qwen3-Coder-Next-DFlash
```

The order of nonblank, non-comment entries defines their roles:

1. The first model is the primary model to serve.
2. The second model is the auxiliary draft model for model-based speculative decoding.
3. Additional models are downloaded but are not automatically selected for serving.

Blank lines and full-line `#` comments do not count toward this order. Downloads
run sequentially and stop on the first failure. Every `/` becomes `--` in the
local folder name. For example, `Qwen/Qwen3-Coder-Next-FP8` is stored in
`models/Qwen--Qwen3-Coder-Next-FP8`.

```bash
bash "$DEPLOY_KIT_ROOT/utility/download_model.sh"
# Or download a custom list:
bash "$DEPLOY_KIT_ROOT/utility/download_model.sh" /path/to/models.txt
```

The downloader uses repository-root `.venv/bin/hf`. If `hf` is missing, install
it with `"$PROJECT/.venv/bin/python" -m pip install huggingface_hub`. For private
or gated repositories, first run `"$PROJECT/.venv/bin/hf" auth login`.

`slurm-vllm.sbatch` reads `$PROJECT/models/models.txt` by default. Set `MODEL_LIST`
to use a custom file; relative paths resolve against `SLURM_SUBMIT_DIR`.
It derives these defaults from the example above:

```text
MODEL_NAME=Qwen--Qwen3-Coder-Next-FP8
MODEL_PATH=$PROJECT/models/Qwen--Qwen3-Coder-Next-FP8
SPEC_MODEL=$PROJECT/models/z-lab--Qwen3-Coder-Next-DFlash
SERVED_MODEL_NAME=Qwen3-Coder-Next-FP8
```

The launcher strips the first `author--` prefix from `MODEL_NAME` for the default
API/UI `SERVED_MODEL_NAME`. Explicit `MODEL_NAME`, `MODEL_PATH`, `SPEC_MODEL`, and
`SERVED_MODEL_NAME` overrides take precedence. `MODEL_REPO` overrides the checkpoint
parent directory; it does not change the default list location.

The batch script uses the second entry for `SPEC_METHOD=dflash` (the default) or
`SPEC_METHOD=eagle3` (EAGLE3). Select the method explicitly to match the auxiliary
checkpoint; the list does not infer it. These methods require a second entry
unless `SPEC_MODEL` is explicitly supplied. `SPEC_METHOD=none` and `ngram` do not
use the second entry and allow a one-model list. Do not export `SPEC_MODEL` for
those methods. DSpark is not currently a supported launcher preset; this list
change does not add a DSpark backend.

Keep the complete downloaded checkpoints at the same paths on every node. The
model list is tracked in Git; downloaded model folders and `.venv/` are ignored.

## 3. Inspect networking and submit a preflight job

Inspect the interfaces on each node:

```bash
bash "$DEPLOY_KIT_ROOT/utility/inference-network.sh" --list
```

The default transport is `rdma`. For ambiguous selection, set `NET_IFACE` and,
if needed, `NET_HCA` to the appropriate device names. For TCP, set
`NET_TRANSPORT=socket`. Global interface overrides assume identical interface
names across nodes; do not export a single node's `NET_LOCAL_IP` to every node.
Normally leave `HEAD_IP` unset so the batch script discovers it on the first
allocated host. `HTTP_IFACE` or `HTTP_CLIENT` can guide the separate HTTP
interface selection.

Start with 128K context, one sequence, no speculation and the auto weight
loader. These explicit overrides avoid the batch script's experimental defaults
of 1M context, eight sequences and DFlash with 15 draft tokens, and the
launcher's `instanttensor` loader.

```bash
export CONTEXT_PROFILE=128k
export MAX_NUM_SEQS=1
export MAX_NUM_BATCHED_TOKENS=8192
export SPEC_METHOD=none
export LOAD_FORMAT=auto
export SERVICE_PORT=8040
mkdir -p "$DEPLOY_KIT_ROOT/logs"

sbatch --export=ALL --time=01:00:00 \
    --chdir="$DEPLOY_KIT_ROOT/logs" \
    "$DEPLOY_KIT_ROOT/dgxspark/slurm-vllm.sbatch" --dry-run
```

Add your site's required `--account`, `--partition` and GPU request (for example,
`--gres=gpu:1` where configured) to both submission commands. The script defaults
to four nodes, one task per node and `--mem=0` (all schedulable host memory).
It assumes one visible GPU per node. Overriding `--nodes` changes the default
TP size; the model and memory capacity must support the resulting configuration.

Dry-run still uses an allocation and checks CUDA, model configuration, network
selection, UI imports and local port availability. It prints commands without
loading weights or starting vLLM. It does not validate vLLM CLI support, model
capacity, distributed communication or external reachability. Check that your
installed build supports the printed options before a full run.

## 4. Start serving and inspect logs

Keep the exports above in the same shell, then submit without `--dry-run`:

```bash
sbatch --export=ALL --time=01:00:00 \
    --chdir="$DEPLOY_KIT_ROOT/logs" \
    "$DEPLOY_KIT_ROOT/dgxspark/slurm-vllm.sbatch"
```

Use the job ID printed by `sbatch`:

```bash
JOB_ID=12345                     # replace with your job ID
squeue -j "$JOB_ID"
tail -f "$DEPLOY_KIT_ROOT/logs/vllm-run-$JOB_ID.out"
```

| File | Contents |
| --- | --- |
| `logs/vllm-run-JOBID.out` | Batch diagnostics, head address and host order |
| `logs/vllm-run-JOBID/hosts.txt` | Shared ordered allocation membership |
| `logs/vllm-run-JOBID/node-HOSTNAME.log` | Each node's launcher and vLLM output |

Read the first host's node log for `HTTP candidate:` and vLLM startup messages.
The candidate is printed before vLLM is ready. `HEAD_IP:8041` is the engine
rendezvous endpoint, not the HTTP API. Compare configuration fingerprints across
node logs; there is no automatic remote consistency check.

Without `--chdir`, logs go into the submission directory. When submitting from
outside the checkout, retain the absolute `DEPLOY_KIT_ROOT` export above.

## 5. Test the API and open the UI

From the intended client, substitute the HTTP address printed in the head log:

```bash
BASE_URL=http://192.168.1.10:8040   # replace with the reported HTTP endpoint
curl --noproxy '*' "$BASE_URL/health"
curl --noproxy '*' "$BASE_URL/v1/models"
curl --noproxy '*' "$BASE_URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3-Coder-Next-FP8","messages":[{"role":"user","content":"Write a Python hello-world program."}],"max_tokens":128}'
```

Open `<BASE_URL>/ui/` for the bundled chat page. It is enabled by default; export
`VLLM_UI_ENABLE=0` before submission to disable it. Leave `VLLM_PUBLIC_BASE_URL`
unset for browser same-origin requests, or set it to the browser-facing server
root without `/v1` when using a proxy.

If you export `VLLM_API_KEY` before submission, add
`-H "Authorization: Bearer $VLLM_API_KEY"` to the API requests and configure your
client with the key. A bindable port does not establish remote reachability:
the client must have a route and permitted access to the HTTP address. Keep the
service on a trusted network or behind your authenticated gateway.

The allocation stays active while vLLM serves. Stop it with:

```bash
scancel "$JOB_ID"
```

The site's time limit also stops serving. After validating the baseline, adjust
context, concurrency, transport and speculation individually. Extended context
and FP8/DFlash combinations have not been validated on DGX hardware by this
update.

## Other scripts

`manual/run_vllm.sh` and `manual/stop_vllm.sh` are separate manual launch scripts;
inspect their path and host settings before use. For the Slurm workflow above,
stop with `scancel`. `mpi-smoke.c` and `mpi-smoke.sbatch` provide a separate MPI
test; the serving job uses `srun --mpi=none`.

## Context-length validation and UI environment warnings

If vLLM rejects `max_model_len=1048576` against a derived limit of `262144`,
the request exceeds the checkpoint's native window. For extended profiles, the
launcher now defaults `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` alongside its YaRN
configuration. This opt-in only permits the length check; it does not establish
long-context quality, GPU memory capacity, or draft-model context support.
An explicit value of `0` is preserved. Native-length profiles do not set this flag.
For a native-context baseline, use `CONTEXT_PROFILE=256k SPEC_METHOD=none` and
ensure no conflicting `MAX_MODEL_LEN`, `SPEC_MODEL`, or `SPEC_TOKENS` overrides
remain in your submission environment.

The launcher still accepts the documented `VLLM_UI_*`, `VLLM_MIDDLEWARE_DIR`, and
`VLLM_PUBLIC_BASE_URL` inputs. It passes middleware settings as `TACC_UI_DIR`,
`TACC_UI_PAGE`, `TACC_UI_MODEL`, and `TACC_PUBLIC_BASE_URL` and removes the export
attribute from the project-owned `VLLM_*` variables before starting vLLM. Update
both the launcher and middleware together. This avoids vLLM's unknown-variable
warnings without changing the public UI routes.
