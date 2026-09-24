# Serve Qwen3-Coder-Next on DGX Spark with Slurm

The batch script requests four nodes with one GPU per node and starts one
launcher per node. The primary model defaults to TP=2, PP=2: each pipeline
stage has two tensor ranks across pairs of nodes. Speculative decoding is off
by default, matching `launch-vllm.sh`. The batch script retains its higher
concurrency defaults.
The first allocated node runs the HTTP API and web UI.

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

Use absolute paths for overrides. Relative `NETWORK_SCRIPT`, `MODEL_LIST`,
`MODEL_REPO`, `MODEL_PATH`, `VLLM_UI_DIR` and `VLLM_MIDDLEWARE_DIR` overrides are
resolved against `SLURM_SUBMIT_DIR` by the batch script. For direct launches,
relative file paths use the caller's directory. `SPEC_MODEL` must be an absolute
local directory or a Hugging Face repository ID. `--chdir` controls logs, not
helper discovery. Both scripts check that the helper is readable; no copy or symlink into `dgxspark/` is needed.

## 1. Prepare the environment

Run these commands in Bash, replacing the checkout path:

```bash
cd /absolute/path/to/TACC-vllm-jobscript
export DEPLOY_KIT_ROOT="$PWD"
export PROJECT="$DEPLOY_KIT_ROOT"
bash "$DEPLOY_KIT_ROOT/utility/installer.sh"
```

Run the installer on a Linux DGX node with GPU access under your site's
allocation policy. It currently defaults to Python 3.14 and vLLM 0.30.0,
overridable through `PYTHON_VERSION` and `VLLM_VERSION`. It creates
`.venv/` and `models/` at the installer's repository root and checks CUDA visibility.

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
For the example above, `SPEC_METHOD=dflash` selects these paths and names:

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

The batch script uses the second entry for `SPEC_METHOD=dflash` or
`SPEC_METHOD=eagle3` (EAGLE3). Select the method explicitly to match the auxiliary
checkpoint; the list does not infer it. The default is `SPEC_METHOD=none`, which
loads no draft. Model-based methods require a second entry
unless `SPEC_MODEL` is explicitly supplied. `SPEC_METHOD=none` and `ngram` do not
use the second entry and allow a one-model list. Do not export `SPEC_MODEL` for
those methods. DSpark is not currently a supported launcher preset; this list
change does not add a DSpark backend.

Direct `launch-vllm.sh` runs do not read `models.txt`; supply model overrides
explicitly or use its built-in Qwen checkpoint default.

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
of 1M context and eight sequences, and the launcher's `instanttensor` loader.
Both entry points default to TP=2/PP=2 with speculation off.

```bash
export TP_SIZE=2 PP_SIZE=2
export CONTEXT_PROFILE=128k
export MAX_NUM_SEQS=1
export MAX_NUM_BATCHED_TOKENS=8192
export SPEC_METHOD=none
unset SPEC_MODEL SPEC_TOKENS
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
It assumes one visible GPU per node. The defaults are fixed at TP=2/PP=2;
when overriding `--nodes`, set both `TP_SIZE` and `PP_SIZE` so their product
matches the allocated node count. The batch script rejects mismatches.

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
node logs; there is no automatic remote consistency check. Fingerprints include
runner selection and the extended-context environment switch, but do not compare
weight contents or package versions. The serving step explicitly uses the initial
job working directory, with absolute paths passed to each launcher.

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
root without `/v1` when using a proxy. `/ui/config.json` supplies the public model
name and API-base override; it never includes an API key.

Each assistant response includes a live tok/s chart, retained when streaming
finishes, and min/max/mean/median tok/s statistics. Stop/error responses retain
statistics for the partial output and are labeled partial/failed. Each generated
code block has a **Copy code** button and a language label; **Copy Markdown**
copies the whole response. Copying falls back to an HTTP-compatible mechanism
and then manual selection if the browser denies clipboard access.

The chart uses one-second arrival windows from first output, including reasoning
and pauses; the last window can be shorter. Mean is the arithmetic average of
window rates, and median is the middle rate (or average of the two middle rates).
The top-level tok/s metric remains total output tokens divided by request time,
including TTFT. These metrics measure browser-observed arrivals, not server-only
GPU throughput.

Final and continuous usage requests are enabled by default. The chart uses
server completion-token counts if present and positive with the first output
delta; otherwise it uses an explicitly marked Unicode-characters/4 estimate for
the whole response. A final-only count updates totals, not historical chart
samples. For APIs that reject `stream_options`, turn off both usage checkboxes.
Keys, settings and conversations are kept only in page memory.

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

The launcher accepts the documented `VLLM_UI_*`, `VLLM_MIDDLEWARE_DIR`, and
`VLLM_PUBLIC_BASE_URL` inputs. It passes middleware settings as `TACC_UI_DIR`,
`TACC_UI_PAGE`, `TACC_UI_MODEL`, and `TACC_PUBLIC_BASE_URL` and removes the export
attribute from the project-owned `VLLM_*` variables before starting vLLM.
The middleware accepts legacy `VLLM_*` settings as a fallback, with `TACC_*`
taking precedence. Update both the launcher and middleware together to avoid
vLLM's unknown-variable warnings.

## Primary and draft parallelism

| Setting | Slurm batch default | Direct launcher default |
| --- | --- | --- |
| `TP_SIZE` / `PP_SIZE` | 2 / 2 | 2 / 2 |
| `SPEC_METHOD` | `none` | `none` |
| `SPEC_TP_SIZE` | Target TP with the relay patch enabled; otherwise 1 | Same |
| `CONTEXT_PROFILE` | `1m` | `1m` |
| `MAX_NUM_SEQS` | 8 | 1 above native context; 4 otherwise |
| `MAX_NUM_BATCHED_TOKENS` | 8192 | 4096 above native context; 8192 otherwise |
| `LOAD_FORMAT` | Inherits launcher | `fastsafetensors` with the relay patch enabled; otherwise `instanttensor` |
| `VLLM_UI_ENABLE` | 1 | 1 |

The batch wrapper exports its values to each launcher, so these override the
launcher's own defaults. It exports `SLURM_EXPORT_ENV=ALL` and uses
`srun --export=ALL`; this cannot restore variables discarded by `sbatch --export=NONE`.
Use `--export=ALL` when passing submission-shell settings. Set both TP and PP
when choosing a layout. The batch job enforces `TP * PP == NUM_NODES`; the
launcher also checks `TP * PP == NUM_NODES * visible GPUs per node`.

These examples assume four one-GPU nodes and a shell without stale `SPEC_MODEL`
or `SPEC_TOKENS` overrides. Speculative examples are configurations to validate
on your installed build, not measured performance recommendations.

```bash
# Four pipeline stages, no speculative decoding.
TP_SIZE=1 PP_SIZE=4 SPEC_METHOD=none CONTEXT_PROFILE=128k \
    sbatch "$DEPLOY_KIT_ROOT/dgxspark/slurm-vllm.sbatch"

# N-gram uses no second model; this launcher requires PP=1 and the V1 runner.
TP_SIZE=4 PP_SIZE=1 SPEC_METHOD=ngram VLLM_USE_V2_MODEL_RUNNER=0 \
    CONTEXT_PROFILE=128k sbatch "$DEPLOY_KIT_ROOT/dgxspark/slurm-vllm.sbatch"

# DFlash uses the second model-list entry, with draft TP=1.
TP_SIZE=4 PP_SIZE=1 SPEC_METHOD=dflash SPEC_TP_SIZE=1 \
    CONTEXT_PROFILE=128k sbatch "$DEPLOY_KIT_ROOT/dgxspark/slurm-vllm.sbatch"
```

EAGLE3 requires `SPEC_METHOD=eagle3` and a compatible EAGLE3 checkpoint as the
second entry (or an explicit `SPEC_MODEL`). Do not reuse the DFlash checkpoint
with the EAGLE3 method. Token defaults are n-gram=4, DFlash=15, EAGLE3=3;
`SPEC_TOKENS` overrides them. `none` rejects a leftover `SPEC_MODEL` or
`SPEC_TOKENS`; n-gram rejects `SPEC_MODEL`, PP>1, and explicit V2 selection.

For DFlash/EAGLE3 with target PP>1, the launcher requests Model Runner V2 and
checks the installed build for all of the following before loading weights:

- `Qwen3NextModel` advertises auxiliary hidden-state relay across pipeline stages.
- Draft parallel configuration sets PP=1.
- The V2 runner provides auxiliary-state relay validation.

These checks are stricter than detecting a runner helper alone. A failed check
stops startup with a suggestion to use `SPEC_METHOD=none` or PP=1 with TP equal
to the total GPU count. Passing checks still requires a real inference test.
The draft JSON sets `draft_tensor_parallel_size`; there is no `SPEC_PP_SIZE`
launcher setting. MTP and DSpark remain outside the supported presets.

### What vLLM 0.30.0 changes

Source checked September 24, 2026 against the **v0.30.0 tag**:

- The [release notes](https://github.com/vllm-project/vllm/releases/tag/v0.30.0)
  announce Model Runner V2 PP support for EAGLE3/DFlash/DSpark and MTP.
- [Draft configuration](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/config/speculative.py)
  sets draft PP=1 independently of target PP. Separate target/draft PP is therefore
  supported by the framework.
- [Qwen3NextModel](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/qwen3_next.py)
  still returns only `hidden_states` and `residual` from intermediate PP stages.
  It does not pack or collect the auxiliary states needed across those stages,
  and inherits `supports_aux_hidden_states_over_pp=False` from
  [EagleModelMixin](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/interfaces.py).
- The [V2 runtime validator](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py)
  rejects a target without that capability. `SupportsPP` plus `SupportsEagle3`
  does not bypass this additional requirement.

Consequently, stock v0.30.0 does **not** enable Qwen3-Coder-Next + DFlash/EAGLE3
at TP=1/PP=4 or TP=2/PP=2. The launcher retains its installed-build check so a
future implementation can pass without a hard-coded version block. Do not
enable the capability flag alone: the forward path must implement the relay.
Use TP=4/PP=1 to test drafting on four Sparks, and compare with non-speculative
TP=1/PP=4 and TP=2/PP=2 baselines. These layouts require actual GPU benchmarks;
framework support alone does not predict throughput.

### Experimental local relay patch

The repository includes a reversible, opt-in patch for **exactly vLLM 0.30.0**.
Revision r2 permits positive target TP/PP sizes, including TP=2/PP=2 and PP=1,
subject to vLLM's model divisibility and partitioning constraints and the allocated
GPU count. Draft TP defaults to target TP and must match it in this V2 DFlash
path; draft PP remains 1. A stale `SPEC_TP_SIZE=1` must be unset or changed when
target TP increases. See the
[installation, launch, rollback, and GPU validation guide](../patches/qwen3next-pp/README.md).

Install it with the serving environment's Python using
`utility/qwen3next_pp_patch.py --apply`, once per distinct environment while
jobs are stopped. The launcher only enables it with
`TACC_QWEN3NEXT_PP_DFLASH=1`; it then checks the exact patched source and imported
capability on every node, including with PP=1. `--apply` also upgrades an exact
r1 installation; modified sources are still rejected. The prototype requires
DFlash, V2, eager execution, native context, no sequence parallelism, and no HF
overrides. Its patch ID is included in the configuration fingerprint.

The experimental preset defaults `LOAD_FORMAT=fastsafetensors`: upstream uses
the standard loader for the draft on the last pipeline stage. Explicit `auto`
also works. `instanttensor` is rejected with PP>1 because its world-group draft
loading can wait for ranks that are not loading the draft. Ordinary runs retain
their previous loader default.

CPU relay tests and simulated Slurm handoff pass. The user has reported a
working TP=1/PP=4 run; **r2's combined TP/PP behavior is not yet validated with
real weights on the Sparks**. Recurrent-state restoration, GPU kernels, and
throughput still need the checks in the guide. It is a source patch, not
HTTP middleware or a registered vLLM plugin.

## Startup failures

If `Worker_PP0` reports `GPUModelRunner` has no `drafter`, inspect the effective
speculative method and runner in the node logs. The launcher rejects n-gram with
PP>1 before model startup; setting draft TP=1 does not resolve that combination.
Use the non-speculative baseline above to isolate it.

A missing `ShmRingBuffer.shared_memory` is a different worker-communication
failure. Find the earliest traceback across **all** node logs, before the NCCL
abort and resource-tracker cleanup messages. This script does not patch vLLM's
shared-memory implementation. `--enforce-eager` and longer startup timeouts do
not establish that an unsupported PP/speculation combination can run.

Validation in this checkout covers shell syntax and a simulated Slurm-to-launcher
handoff. Run the regression checks without a GPU:

```bash
bash -n dgxspark/slurm-vllm.sbatch dgxspark/launch-vllm.sh
python3 tests/check_slurm_handoff.py
```

Run from the repository root. If the system Bash is older than 4 (such as the
macOS system Bash), prefix the Python command with
`BASH_BIN=/opt/homebrew/bin/bash` or the path to another Bash 4+ installation.
The checks simulate Slurm, GPU discovery, port binding and the final vLLM
process; embedded configuration generation uses real Python. A real Slurm
allocation is still needed to verify CUDA, networking, model capacity and
inference on DGX Spark.
