#!/usr/bin/env python3
"""Install/check/revert the opt-in vLLM 0.30.0 Qwen3Next DFlash relay patch.

Run with the serving environment's Python. Never run while vLLM is serving.
No vLLM import, dependency installation, or network access is performed.
"""

import argparse
import ast
import hashlib
from importlib.metadata import distribution
import os
from pathlib import Path
import stat
import tempfile

PATCH_ID = "qwen3next-dflash-pp-v030-r2"
LEGACY_PATCH_ID = "qwen3next-dflash-pp-v030-r1"
LEGACY_SHA256 = "bf2c6fbd3797dbc99899b531aac7f65ea23e71a457e2c63226be7a762d3ef186"
UPSTREAM_SHA256 = "4fa5112ac2bf7886d41d9e1f6c3da8793620d1ec49c961b31e9609459a6a7f7a"
MODEL_FILE = "vllm/model_executor/models/qwen3_next.py"

# Small edits to the pinned upstream file. Keep the original PP=1 / opt-out path.
# Derived from vLLM (Apache-2.0); see patches/qwen3next-pp/README.md.
EDITS = (
    ("import torch\n", "import os\n\nimport torch\n"),
    (
        "class Qwen3NextModel(nn.Module, EagleModelMixin):\n",
        "class Qwen3NextModel(nn.Module, EagleModelMixin):\n"
        f'    _tacc_pp_patch = "{PATCH_ID}"\n'
        '    supports_aux_hidden_states_over_pp = (\n'
        '        os.environ.get("TACC_QWEN3NEXT_PP_DFLASH") == "1"\n'
        '    )\n\n',
    ),
    (
        "        config: Qwen3NextConfig = vllm_config.model_config.hf_text_config\n"
        "        parallel_config = vllm_config.parallel_config\n",
        "        config: Qwen3NextConfig = vllm_config.model_config.hf_text_config\n"
        "        parallel_config = vllm_config.parallel_config\n"
        "        if self.supports_aux_hidden_states_over_pp:\n"
        "            spec = vllm_config.speculative_config\n"
        "            if (\n"
        '                config.model_type != "qwen3_next"\n'
        "                or parallel_config.tensor_parallel_size < 1\n"
        "                or parallel_config.pipeline_parallel_size < 1\n"
        "                or parallel_config.use_sequence_parallel_moe\n"
        "                or spec is None\n"
        '                or spec.method != "dflash"\n'
        "                or spec.draft_tensor_parallel_size not in (\n"
        "                    None, parallel_config.tensor_parallel_size\n"
        "                )\n"
        '                or os.environ.get("VLLM_USE_V2_MODEL_RUNNER") != "1"\n'
        "            ):\n"
        "                raise RuntimeError(\n"
        '                    "Experimental Qwen3Next relay requires qwen3_next, "\n'
        '                    "DFlash, V2, TP>=1, PP>=1, draft TP matching target TP, "\n'
        '                    "and no sequence parallelism"\n'
        "                )\n",
    ),
    (
        "        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)\n",
        "        relay_aux = (\n"
        "            self.supports_aux_hidden_states_over_pp\n"
        "            and get_pp_group().world_size > 1\n"
        "        )\n"
        '        if relay_aux and self.config.model_type != "qwen3_next":\n'
        '            raise RuntimeError("Experimental PP relay is scoped to qwen3_next")\n'
        "        remote_aux = (\n"
        "            self.collect_remote_aux_hidden_states(intermediate_tensors)\n"
        "            if relay_aux else []\n"
        "        )\n"
        "        # A boundary tap belongs to the preceding stage. Only stage 0\n"
        "        # captures the embedding tap; never duplicate it on later ranks.\n"
        "        aux_hidden_states = []\n"
        "        if not relay_aux or get_pp_group().is_first_rank:\n"
        "            self._maybe_add_hidden_state(aux_hidden_states, 0, hidden_states, residual)\n",
    ),
    (
        '                {"hidden_states": hidden_states, "residual": residual}\n',
        "                {\n"
        '                    "hidden_states": hidden_states,\n'
        '                    "residual": residual,\n'
        "                    **(self.pack_local_aux_hidden_states(aux_hidden_states)\n"
        "                       if relay_aux else {}),\n"
        "                }\n",
    ),
    (
        "        if aux_hidden_states:\n"
        "            return hidden_states, aux_hidden_states\n",
        "        # V2 relays upstream slots through intermediate ranks; the final\n"
        "        # rank assembles the ordered features consumed by the drafter.\n"
        "        aux_hidden_states = remote_aux + aux_hidden_states\n"
        "        if aux_hidden_states:\n"
        "            return hidden_states, aux_hidden_states\n",
    ),
)

# Only the marker and constructor guard changed from r1. Reconstruct its exact
# edits for a hash-checked migration; never accept arbitrary previously edited code.
LEGACY_EDITS = tuple(
    (before, after.replace(PATCH_ID, LEGACY_PATCH_ID)
     .replace("parallel_config.tensor_parallel_size < 1",
              "parallel_config.tensor_parallel_size != 1")
     .replace("parallel_config.pipeline_parallel_size < 1",
              "parallel_config.pipeline_parallel_size not in (1, 2, 4)")
     .replace("                or spec.draft_tensor_parallel_size not in (\n"
              "                    None, parallel_config.tensor_parallel_size\n"
              "                )\n", "")
     .replace('"DFlash, V2, TP>=1, PP>=1, draft TP matching target TP, "\n'
              '                    "and no sequence parallelism"',
              '"DFlash, V2, TP=1, PP=1/2/4, and no sequence parallelism"'))
    for before, after in EDITS
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def transform(data, reverse=False, edits=EDITS):
    text = data.decode("utf-8")
    for before, after in (reversed(edits) if reverse else edits):
        old, new = (after, before) if reverse else (before, after)
        if text.count(old) != 1:
            raise ValueError("Source does not match the exact patch anchors; refusing changes")
        text = text.replace(old, new, 1)
    ast.parse(text)
    return text.encode("utf-8")


def source_state(data):
    if digest(data) == UPSTREAM_SHA256:
        return "upstream"
    if digest(data) == LEGACY_SHA256:
        return "patched-r1"
    try:
        restored = transform(data, reverse=True)
    except (ValueError, SyntaxError, UnicodeError):
        return "unknown"
    return "patched" if digest(restored) == UPSTREAM_SHA256 else "unknown"


def installed_source():
    dist = distribution("vllm")
    if dist.version != "0.30.0":
        raise ValueError(f"Requires exactly vllm==0.30.0; found {dist.version}")
    path = Path(dist.locate_file(MODEL_FILE)).resolve()
    if not path.is_file():
        raise ValueError(f"Cannot find installed source: {path}")
    return path


def update_source(path, action):
    """Atomic, idempotent edits; refuse modified sources, including on revert."""
    path = Path(path)
    if action in ("check", "status"):
        state = source_state(path.read_bytes())
        if state == "unknown":
            raise ValueError("Installed source differs from the pinned upstream/patch; refusing changes")
        if action == "check" and state != "patched":
            if state == "patched-r1":
                raise ValueError("Patch r1 is installed; run --apply to upgrade to r2")
            raise ValueError("Patch is not installed; run this tool with --apply first")
        return state
    lock = path.with_name(path.name + ".tacc-patch.lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    temporary = None
    try:
        data = path.read_bytes()
        state = source_state(data)
        if state == "unknown":
            raise ValueError("Installed source differs from the pinned upstream/patch; refusing changes")
        desired = "patched" if action == "apply" else "upstream"
        if state == desired:
            return state
        if state == "patched-r1":
            upstream = transform(data, reverse=True, edits=LEGACY_EDITS)
            if digest(upstream) != UPSTREAM_SHA256:
                raise ValueError("Legacy patch restoration failed")
            result = transform(upstream) if action == "apply" else upstream
        else:
            result = transform(data, reverse=action == "revert")
        if source_state(result) != desired:
            raise ValueError("Patch verification failed")
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
            temporary = Path(out.name)
            out.write(result)
            out.flush()
            os.fsync(out.fileno())
        temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        # The lock protects cooperating invocations; also detect external edits.
        if path.read_bytes() != data:
            raise ValueError("Source changed during patching; refusing replacement")
        os.replace(temporary, path)
        temporary = None
        return desired
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for action in ("apply", "check", "status", "revert"):
        group.add_argument(f"--{action}", dest="action", action="store_const", const=action)
    args = parser.parse_args()
    try:
        path = installed_source()
        state = update_source(path, args.action)
    except (OSError, ValueError, LookupError, ImportError) as exc:
        parser.exit(1, f"Qwen3Next patch: {exc}\n")
    print(f"{PATCH_ID}: {state}; {path}")


if __name__ == "__main__":
    main()
