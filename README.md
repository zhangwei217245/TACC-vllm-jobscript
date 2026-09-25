# vLLM serving on DGX Spark

Launch Qwen3-Coder-Next-FP8 across four DGX Spark nodes through Slurm, with an
OpenAI-compatible API and a browser chat UI on the first allocated node.
The current baseline uses **TP=2, PP=2, speculative decoding off**. Each node
runs one launcher with one visible GPU. Set both `TP_SIZE` and `PP_SIZE` to
choose another topology; their product must equal the allocated GPU count.

See the [DGX Spark deployment guide](dgxspark/README.md) for the full setup,
network configuration, overrides and troubleshooting. The launcher remains
Qwen3-Coder-Next-specific: changing the model list does not automatically adapt
RoPE scaling, tool parsing or speculative-decoding compatibility to another model.

## Project layout

```text
TACC-vllm-jobscript/
├── dgxspark/
│   ├── slurm-vllm.sbatch
│   └── launch-vllm.sh
├── models/
│   └── models.txt          # first entry: primary; second: optional draft
├── utility/
│   ├── installer.sh
│   ├── inference-network.sh
│   └── download_model.sh
├── vllm_middleware/static_ui.py
├── .venv/                  # created by installer.sh
└── ui/chat.html
```

The installer and downloader use the repository root regardless of the current
working directory. Their destinations are `.venv/` and `models/`. The serving
scripts default `PROJECT` to this root; overriding it selects another existing
environment/model directory and does not redirect the installer or downloader.
All nodes need the same absolute paths. The job log directory must be shared
because it contains the generated host list.

## Install and download

Run on Linux with GPU access according to your site's allocation policy:

```bash
cd /absolute/path/to/TACC-vllm-jobscript
export DEPLOY_KIT_ROOT="$PWD"
bash utility/installer.sh
bash utility/download_model.sh
```

The installer currently defaults to Python **3.14** and vLLM **0.30.0**.
`PYTHON_VERSION` and `VLLM_VERSION` override these values. An existing `.venv`
with a different Python version is rejected rather than replaced.

Edit [models/models.txt](models/models.txt) before downloading:

```text
Qwen/Qwen3-Coder-Next-FP8
z-lab/Qwen3-Coder-Next-DFlash
```

Use one `organization/model` ID per line. Blank lines and full-line `#` comments
are ignored. The first entry selects the primary model. The second selects the
draft only when `SPEC_METHOD=dflash` or `eagle3`. Later entries are downloaded
but not automatically served. Downloads are sequential and stop on failure.
Every `/` becomes `--` in the local folder name.

For the example above, the primary folder is
`models/Qwen--Qwen3-Coder-Next-FP8`, and the public model name is
`Qwen3-Coder-Next-FP8` (the first `author--` prefix is stripped). The default
`SPEC_METHOD=none` needs only the first entry. `ngram` also uses no draft model.
A second entry alone never enables speculative decoding.

The batch script reads `$PROJECT/models/models.txt`. `MODEL_LIST`, `MODEL_REPO`,
`MODEL_NAME`, `MODEL_PATH`, `SPEC_MODEL` and `SERVED_MODEL_NAME` can override the
selection. `MODEL_REPO` changes checkpoint storage, not the model-list location.
For a custom download list, run `bash utility/download_model.sh /path/to/list.txt`.
Direct launcher runs do not read the list; model overrides must be supplied explicitly.

## Submit a baseline job

The batch defaults retain an experimental 1M context and eight concurrent
sequences. Start with the smaller baseline below. Add your site's required
account, partition and GPU request, such as `--gres=gpu:1` where configured.

```bash
export TP_SIZE=2 PP_SIZE=2 SPEC_METHOD=none
unset SPEC_MODEL SPEC_TOKENS
export CONTEXT_PROFILE=128k MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=8192
export LOAD_FORMAT=auto SERVICE_PORT=8040
mkdir -p "$DEPLOY_KIT_ROOT/logs"

# Allocate nodes, run preflight checks, and print commands without loading weights.
sbatch --export=ALL --time=01:00:00 --chdir="$DEPLOY_KIT_ROOT/logs" \
    "$DEPLOY_KIT_ROOT/dgxspark/slurm-vllm.sbatch" --dry-run

# Start serving with the same environment.
sbatch --export=ALL --time=01:00:00 --chdir="$DEPLOY_KIT_ROOT/logs" \
    "$DEPLOY_KIT_ROOT/dgxspark/slurm-vllm.sbatch"
```

Slurm selects the host order, discovers the head's engine IP, and invokes the
launcher once on every node. The launcher reads the host list to determine its
rank, activates `$PROJECT/.venv`, and starts vLLM. Rank 0 provides HTTP/UI;
the remaining ranks use `--headless`. The executor is `mp`, with `srun --mpi=none`.

Logs are in `logs/vllm-run-JOBID.out` and `logs/vllm-run-JOBID/node-HOSTNAME.log`.
The head log prints the HTTP candidate and UI URL; wait for vLLM startup to finish.
`HEAD_IP:8041` is the engine rendezvous endpoint, not the browser/API endpoint.
`--chdir` controls the log location; without it, logs use the submission directory.
Stop serving with `scancel JOBID`.

## Parallelism and speculation

| Setting | Slurm default | Direct launcher default |
| --- | --- | --- |
| Primary TP / PP | 2 / 2 | 2 / 2 |
| Speculative method | `none` | `none` |
| Context | `1m` (1,048,576 tokens) | `1m` |
| Maximum sequences | 8 | 1 above native context; 4 otherwise |
| Batch token budget | 8192 | 4096 above native context; 8192 otherwise |
| Weight loader | `instanttensor` | `instanttensor` |
| UI | Enabled on rank 0 | Enabled on rank 0 |

For four one-GPU nodes, `TP_SIZE=1 PP_SIZE=4 SPEC_METHOD=none` is another explicit
layout. A different node count requires explicit TP/PP values; defaults do not
resize automatically. Exported submission settings take precedence.

The launcher restricts n-gram speculation to PP=1 and rejects an explicit V2
runner for that preset. DFlash/EAGLE3 with PP>1 require an installed build with
Qwen3Next auxiliary hidden-state relay, a compatible V2 runner and draft PP=1;
the launcher checks these before loading weights. Changing the draft TP/PP alone
does not satisfy those requirements. See the [speculation examples](dgxspark/README.md#primary-and-draft-parallelism).
MTP and DSpark are not implemented as launcher presets.

**vLLM 0.30.0 distinction (source checked September 24, 2026):** the
[release adds PP speculation in Model Runner V2](https://github.com/vllm-project/vllm/releases/tag/v0.30.0),
but the tagged Qwen3Next implementation still lacks auxiliary-state relay across
PP stages. `SupportsPP` and `SupportsEagle3` individually do not establish support
for their combination. For stock 0.30.0, use PP=1 for Qwen3-Coder-Next with
DFlash/EAGLE3; see the [model-specific evidence](dgxspark/README.md#primary-and-draft-parallelism).

An opt-in [Qwen3Next PP+DFlash prototype](patches/qwen3next-pp/README.md) now
provides a reversible patch for exactly vLLM 0.30.0. It implements the missing
relay for experimental target **TP>=1/PP>=1** runs, including TP=2/PP=2.
Draft TP must match target TP in this V2 path; draft PP remains 1. Model
divisibility, layer partitioning, and GPU-allocation constraints still apply.
The preset defaults to `fastsafetensors` and rejects `instanttensor` with PP>1.
CPU tests and simulated handoff pass; combined TP/PP GPU correctness and
performance remain unverified. It is disabled by default and requires explicit
installation and `TACC_QWEN3NEXT_PP_DFLASH=1`. Existing r1 installations upgrade
with the same patch tool's `--apply` command; see the guide above.

## Browser chat UI

Open `<HTTP_BASE_URL>/ui/`. The page fetches `/ui/config.json` to prefill the
API URL and served model name. By default it uses the browser's origin, including
SSH tunnels and proxy prefixes. Set `VLLM_PUBLIC_BASE_URL` to a public server root
without `/v1` to override this, or `VLLM_UI_ENABLE=0` to disable the UI.

The updated UI provides:

- Streaming Markdown and a **Copy code** button attached to each code block.
  A separate **Copy Markdown** button copies the whole response.
- A live tok/s chart in each assistant response, retained after completion.
- Live violin and box plots below the monitor, plus **p50, p80, p90, p95, p99,
  p99.9 and p99.99** for the response's complete one-second windows.
- A rolling high-speed frequency plot comparing the latest 30 complete windows
  with the preceding 30. Set a tok/s threshold to see the percentage at or above
  that speed, and the change in percentage points between groups.
- **Min, max, mean and median tok/s** at the end of each response, including
  partial responses when stopped or interrupted.
- Client-observed TTFT, output-token totals and an end-to-end average rate above
  the conversation. These are distinct from the per-window chart rates.

Charts measure arrivals in one-second windows from first output, including
reasoning and pauses; the final line-chart window may be shorter. Summary
statistics and distribution plots exclude partial windows so that median and p50
agree. Responses shorter than one second have no distribution yet. Percentiles
use linear interpolation; p99.9 and p99.99 on short responses mostly describe
the maximum, with too few windows to characterize rare speeds. The violin uses
smoothed density; box whiskers use 1.5×IQR, with outlier dots thinned above 80.
The rolling frequency view is the most direct way to track how often a chosen
high speed occurs. Both final and continuous usage are
requested by default. If the first output delta includes a positive server
completion-token count, the chart uses server counts. Otherwise it remains an
explicitly labeled characters-divided-by-four estimate for that response.
Final-only usage can correct totals, but cannot reconstruct the live chart.
Network buffering affects the measurements; these are not server-only decode rates.

Run `node tests/check_chat_rates.cjs` for the window, percentile, distribution,
and SVG checks. The plots use inline SVG with no additional network dependencies.

For APIs that reject stream options, disable both usage checkboxes. Code copying
tries the Clipboard API, then a fallback for HTTP deployments, then manual
selection if the browser refuses copying. No conversations, settings or API keys
are saved by the page. When `VLLM_API_KEY` is configured, enter it in the UI or
send `Authorization: Bearer ...` with API requests.

The launcher accepts the public `VLLM_UI_*` settings and passes middleware values
as `TACC_UI_*` / `TACC_PUBLIC_BASE_URL` to avoid vLLM unknown-variable warnings.
The middleware also accepts legacy names, with `TACC_*` taking precedence.
