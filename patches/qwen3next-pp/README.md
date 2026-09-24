# Experimental Qwen3-Coder-Next PP + DFlash relay

This is a reversible **vLLM 0.30.0 source patch**, not HTTP middleware or a
registered model plugin. It implements Qwen3Next auxiliary hidden-state relay
using vLLM's existing Model Runner V2 helpers. It is inactive unless
`TACC_QWEN3NEXT_PP_DFLASH=1` is set before starting Python.

**Revision r2: CPU relay tests and simulated Slurm handoff pass. A user-reported
TP=1/PP=4 run works, but combined TP/PP GPU inference, recurrent-state correctness,
and performance remain unvalidated.** Use an experimental job first.

Target **TP>=1 and PP>=1** are accepted, including TP=2/PP=2, TP=4/PP=1,
and TP=1/PP=4. This removes the patch's topology whitelist; it does not override
vLLM's model head/weight divisibility, layer partitioning, or hardware constraints.
For the Slurm script, TP times PP must equal the number of one-GPU nodes.

**Draft TP must match target TP; draft PP stays 1.** In the tagged V2 DFlash
loader, the draft uses the current target TP group, including its sharded
embedding and output head. This patch does not implement separate draft TP
groups. `SPEC_TP_SIZE` defaults to `TP_SIZE` when opted in; an explicit mismatch
is rejected. Unset stale `SPEC_TP_SIZE=1` exports when increasing target TP.

The launcher requires DFlash, V2 (even at PP=1), eager execution, an unextended
context, and no HF overrides. It defaults eager execution on when opted in.
Sequence parallelism and other model types remain rejected. This does not add
DSpark/EAGLE3 presets or change any quantization.

## What changes

The [reviewable diff](vllm-0.30.0.patch) changes only
`vllm/model_executor/models/qwen3_next.py`:

1. Advertise relay capability only when explicitly enabled.
2. Capture the embedding feature only on the first stage. A feature at a stage
   boundary belongs to the preceding stage, avoiding duplicate captures.
3. Pack each stage's local auxiliary states into its `IntermediateTensors`.
4. Let V2 relay upstream slots across intervening stages, then collect them
   before local features on the final stage.

The existing residual reconstruction, model weights, draft implementation,
sampling/verification, and recurrent-state machinery are reused. Reusing that
machinery does not establish that it is correct for this new combination.
The opt-out forward behavior and PP=1 feature behavior are covered by CPU tests.

Without sequence parallelism, auxiliary capture points contain full hidden
states on each TP rank. The existing PP transport relays those tensors between
corresponding TP ranks; its optional slice/all-gather optimization reconstructs
them on the receiving stage. No extra all-reduce or new process group is added
by this patch. See the tagged [TP/PP transport](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/distributed/parallel_state.py),
[DFlash loader](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/worker/gpu/spec_decode/dflash/utils.py),
and [draft model](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/model_executor/models/qwen3_dflash.py).

## Install and check

Stop jobs using the environment before applying or reverting the patch. Run
from the repository root using the **serving environment's Python**:

```bash
.venv/bin/python utility/qwen3next_pp_patch.py --status
.venv/bin/python utility/qwen3next_pp_patch.py --apply
.venv/bin/python utility/qwen3next_pp_patch.py --check
```

Apply once per distinct environment. A shared `.venv` needs one application;
node-local environments need the same patch on every node. The tool requires
exactly `vllm==0.30.0` and the SHA-256 of the tagged upstream source, or the
exact already-patched source. An exact r1 installation is recognized as
`patched-r1`: `--apply` upgrades it atomically to r2, `--check` asks for the
upgrade, and `--revert` restores upstream from either revision. It refuses
locally modified files and different versions. Applying twice is harmless;
writes use a lock and atomic replacement.
`--check` is read-only and is safe for concurrent startup checks.

There are no downloads or dependency changes in this tool. A vLLM reinstall
can remove the patch; check again after rebuilding the environment.

## Launch a four-node experiment

Ensure `models/models.txt` selects your Qwen3-Coder-Next checkpoint first and
`z-lab/Qwen3-Coder-Next-DFlash` second, and both are downloaded. From the
repository root, in Bash:

```bash
export DEPLOY_KIT_ROOT="$PWD"
export PROJECT="$DEPLOY_KIT_ROOT"
unset SPEC_MODEL SPEC_TOKENS SPEC_TP_SIZE HF_OVERRIDES
export TACC_QWEN3NEXT_PP_DFLASH=1
export VLLM_USE_V2_MODEL_RUNNER=1
export TP_SIZE=2 PP_SIZE=2 SPEC_METHOD=dflash
export SPEC_TOKENS=3
export CONTEXT_PROFILE=128k MAX_MODEL_LEN=8192
export MAX_NUM_SEQS=1 MAX_NUM_BATCHED_TOKENS=8192
export ENFORCE_EAGER=1 PREFIX_CACHING=0
export LOAD_FORMAT=fastsafetensors

# Runs preflight on all allocated nodes; does not load weights or run inference.
sbatch --export=ALL dgxspark/slurm-vllm.sbatch --dry-run

# Submit after checking the dry-run job logs.
sbatch --export=ALL dgxspark/slurm-vllm.sbatch
```

The short context, single request, and disabled prefix caching isolate initial
correctness testing. The example uses three speculative tokens as a benchmark
starting point; the launcher default remains 15. Keep the count fixed while
comparing layouts. On four nodes, compare `TP_SIZE=1 PP_SIZE=4`,
`TP_SIZE=2 PP_SIZE=2`, and `TP_SIZE=4 PP_SIZE=1`; let draft TP follow target TP.
For a two-node test use `TP_SIZE=1 PP_SIZE=2` with `sbatch --nodes=2`.
The ordinary four-node defaults remain TP=2/PP=2 with speculation off when
these exported overrides are absent.

The experimental loader defaults to `fastsafetensors`. Upstream falls back to
the standard loader for the draft at PP>1, avoiding world collectives on stages
that do not load a draft. `LOAD_FORMAT=auto` is another option. The launcher
rejects `instanttensor` with experimental PP>1: its world-group draft loading
can deadlock. The patch does not change upstream loader code.

Every launcher checks the installed source before weight loading, verifies the
imported patch marker/capability, and includes the patch ID in its configuration
fingerprint. Compare fingerprints across node logs. V2 determines draft PP=1;
there is no `SPEC_PP_SIZE` option.

## Validation required on the cluster

Local checks (Bash 4+ required for the handoff test):

```bash
python3 -m unittest discover -s tests -p test_qwen3next_pp_patch.py -v
python3 tests/check_slurm_handoff.py
```

The relay tests execute the actual patched forward method with deterministic
scalar layers and the pinned upstream `EagleModelMixin` and V2 relay method.
They compare 2,100 partition/tap combinations against the unpartitioned
upstream reference, including uneven partitions, boundary/embedding/final taps,
and stages with no selected features. They also check missing transport,
opt-out behavior, scope guards, version refusal, and r1 upgrade/apply/check/revert.
With PyTorch installed, an additional CPU tensor test covers 180 TP/PP/tap/token
combinations using simulated sharded layers and PP slice/all-gather transport.
It checks reconstructed feature values, shapes, residuals, and ordering on each
TP lane. It is skipped if PyTorch is unavailable. These tests do not execute
vLLM GPU kernels, NCCL, the real DFlash model, or recurrent caches.
The handoff test stubs vLLM imports/configuration in its
experimental cases; it tests argument/environment propagation, not vLLM startup.

Before treating this combination as supported:

1. Establish a working PP=1 DFlash reference with the same target/draft weights,
   tokenizer, dtype, and short prompts on a topology where the weights fit.
2. Compare real auxiliary tensors at each TP/PP layout against the PP=1 reference:
   values within numerical tolerance, tap order, shapes, dtype, and token alignment.
   Check every TP lane and shared embedding/output-head shards on the final stage.
3. Compare target verification logits with speculation off, including forced
   draft rejections, partial acceptance, EOS, and multiple sequential requests.
   Check recurrent-state restoration, not only whether generated text looks plausible.
4. Exercise mixed prompt lengths, chunked prefill, request cancellation,
   concurrency, and prefix caching separately. Check for stale auxiliary buffers
   and recurrent states. Graph mode needs separate work and is currently rejected
   by the experimental launcher path.
5. Only after correctness passes, measure acceptance length, output tok/s,
   TTFT, and memory against the same non-speculative topology and TP4/PP1.
   Keep context, prompts, sampling, and concurrency consistent.

Additional V2/hybrid-model failures may require changes outside this relay
patch. Removing capability checks or advertising the flag alone is insufficient.

## Disable or revert

Stop the experimental job. To return to an ordinary baseline in the same shell:

```bash
unset TACC_QWEN3NEXT_PP_DFLASH VLLM_USE_V2_MODEL_RUNNER SPEC_MODEL SPEC_TOKENS
export SPEC_METHOD=none
# Choose TP_SIZE/PP_SIZE explicitly for the desired baseline.
```

The installed patch is then inactive in new processes. To restore the exact
upstream source (after stopping all jobs using this environment):

```bash
.venv/bin/python utility/qwen3next_pp_patch.py --revert
```

Revert refuses unexpected edits rather than overwriting them. Process-local
imports do not change until those processes restart.

## Provenance

Base tag: [vLLM v0.30.0](https://github.com/vllm-project/vllm/tree/v0.30.0).
Upstream `qwen3_next.py` SHA-256:
`4fa5112ac2bf7886d41d9e1f6c3da8793620d1ec49c961b31e9609459a6a7f7a`.

The patch, edit snippets, and files under `tests/fixtures/qwen3next_pp` derive
from vLLM, copyright its contributors, licensed under
[Apache-2.0](LICENSE-vllm). The full upstream model source is retained as an
offline fixture; the mixin and relay fixtures are extracted verbatim from
`model_executor/models/interfaces.py` and `v1/worker/gpu/pp_utils.py`.
Our additions are the opt-in guards, model relay integration, patch management,
launcher integration, and tests. The diff records the changes to the original.
