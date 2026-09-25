#!/usr/bin/env python3
"""Run AIPerf or index its JSON exports. Standard library only; Python >=3.9."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = "profile_export_aiperf.json"
SERVER_ENV = ("TP_SIZE", "PP_SIZE", "SPEC_METHOD", "SPEC_MODEL", "SPEC_TP_SIZE",
              "SPEC_TOKENS", "MAX_MODEL_LEN", "MAX_NUM_SEQS", "MAX_NUM_BATCHED_TOKENS",
              "TACC_QWEN3NEXT_PP_DFLASH", "VLLM_USE_V2_MODEL_RUNNER", "LOAD_FORMAT")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def read_json(path):
    with Path(path).open() as f:
        value = json.load(f, parse_constant=lambda v: None)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def model_defaults():
    models = ROOT / "models/models.txt"
    if not models.exists():
        return None, None
    first = next((line.strip() for line in models.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith("#")), None)
    if not first:
        return None, None
    local = ROOT / "models" / first.replace("/", "--")
    return first.split("/", 1)[-1], str(local) if local.is_dir() else first


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def benchmark(args):
    if not args.model or not args.tokenizer:
        raise ValueError("Specify --model (served API name) and --tokenizer (HF ID or local directory).")
    parsed = urlsplit(args.url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("--url must be an HTTP(S) server URL without credentials, query, or fragment")
    if not args.endpoint.startswith("/") or "?" in args.endpoint or "#" in args.endpoint:
        raise ValueError("--endpoint must be a path such as /v1/chat/completions")
    if args.requests < args.concurrency:
        raise ValueError("--requests must be at least --concurrency")
    url = args.url.rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    executable = Path(os.environ.get("AIPERF_BIN", ROOT / ".venv-aiperf/bin/aiperf")).expanduser().resolve()
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise ValueError("AIPerf is not installed. Run utility/installer.sh, or set AIPERF_BIN to its executable.")
    version = subprocess.check_output([str(executable), "--version"], text=True, timeout=30).strip()
    out = args.root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    label = re.sub(r"[^a-zA-Z0-9_.-]+", "-", args.label).strip(".-")[:60] or "benchmark"
    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + label + "-"
    run_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=out))
    command = [str(executable), "profile", "--model", args.model, "--tokenizer", args.tokenizer,
               "--url", url, "--endpoint-type", "chat", "--endpoint", args.endpoint,
               "--streaming", "--concurrency", str(args.concurrency), "--request-count", str(args.requests),
               "--warmup-request-count", str(args.warmup), "--synthetic-input-tokens-mean", str(args.input_tokens),
               "--synthetic-input-tokens-stddev", "0", "--output-tokens-mean", str(args.output_tokens),
               "--output-tokens-stddev", "0", "--random-seed", str(args.seed),
               "--request-timeout-seconds", str(args.timeout), "--ui", "simple",
               "--extra-inputs", json.dumps({"temperature": 0, "ignore_eos": not args.allow_eos}),
               "--output-artifact-dir", str(run_dir)]
    if args.request_rate is not None:
        command += ["--request-rate", str(args.request_rate)]
    env = os.environ.copy()
    key = env.get("VLLM_API_KEY") or env.get("OPENAI_API_KEY")
    if key:
        # AIPerf resolves this reference internally. No secret in argv/run.json.
        env["AIPERF_BENCH_API_KEY"] = key
        command += ["--api-key", "${AIPERF_BENCH_API_KEY}"]
    metadata = {"schema_version": 1, "run_id": run_dir.name, "label": args.label,
                "started_at": now(), "status": "running", "aiperf_version": version,
                "url": url, "endpoint": args.endpoint, "model": args.model, "tokenizer": args.tokenizer,
                "workload": {k: getattr(args, k) for k in ("concurrency", "requests", "warmup", "input_tokens", "output_tokens", "seed", "request_rate", "timeout", "allow_eos")},
                "server_job_id": args.server_job_id,
                "server_settings_unverified": {k: env[k] for k in SERVER_ENV if k in env},
                "command": command, "summary_file": SUMMARY, "log_file": "aiperf.log"}
    atomic_json(run_dir / "run.json", metadata)
    print(f"Run: {run_dir}\nProgress log: {run_dir / 'aiperf.log'}", flush=True)
    code = 1
    process = None
    try:
        with (run_dir / "aiperf.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
            code = process.wait()
        metadata["status"] = "completed" if code == 0 else "failed"
        if code == 0:
            read_json(run_dir / SUMMARY)  # Success must include a usable JSON export.
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        code = 130
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    except (OSError, ValueError) as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        code = code or 2
    finally:
        metadata.update(finished_at=now(), exit_code=code)
        atomic_json(run_dir / "run.json", metadata)
    print(f"{metadata['status']}: {run_dir / 'run.json'}", flush=True)
    return code if code >= 0 else 128 - code


def metric_columns(summary):
    """Preserve units/stat names rather than mixing latency and throughput."""
    columns = {}
    for metric, block in summary.items():
        if not isinstance(block, dict) or not isinstance(block.get("unit"), str):
            continue
        for stat, value in block.items():
            if stat == "unit" or (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)):
                columns[f"{metric}.{stat}"] = value
    return columns


def report(args):
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Output directory not found: {root}")
    manifests = set(root.rglob("run.json"))
    directories = {p.parent for p in manifests} | {p.parent for p in root.rglob(SUMMARY)}
    rows, entries = [], []
    for directory in sorted(directories):
        # Wrapper runs own their native phase/sub-run artifacts. Do not count both.
        if any((parent / "run.json") in manifests for parent in directory.parents if parent != root.parent):
            continue
        errors, meta, summary = [], {}, {}
        manifest = directory / "run.json"
        if manifest.exists():
            try:
                meta = read_json(manifest)
            except (ValueError, OSError) as exc:
                errors.append(f"Invalid run.json: {exc}")
        try:
            summary = read_json(directory / SUMMARY)
        except (ValueError, OSError) as exc:
            errors.append(f"Summary unavailable: {exc}")
        workload = meta.get("workload") or {}
        server = meta.get("server_settings_unverified") or {}
        if not isinstance(workload, dict): workload = {}
        if not isinstance(server, dict): server = {}
        row = {"run_dir": str(directory.relative_to(root)), "label": meta.get("label", ""),
               "status": meta.get("status", "imported" if summary else "invalid"),
               "started_at": meta.get("started_at", ""), "finished_at": meta.get("finished_at", ""),
               "exit_code": meta.get("exit_code", ""), "model": meta.get("model", ""),
               "url": meta.get("url", ""), "aiperf_version": meta.get("aiperf_version", ""),
               "aiperf_schema_version": summary.get("schema_version", ""),
               "server_job_id": meta.get("server_job_id", ""),
               "errors": "; ".join(errors)}
        row.update({f"workload.{k}": v for k, v in workload.items() if not isinstance(v, (list, dict))})
        row.update({f"server_unverified.{k}": v for k, v in server.items() if k in SERVER_ENV})
        row.update(metric_columns(summary))
        rows.append(row)
        entries.append({**row, "manifest": str(manifest.relative_to(root)) if manifest.exists() else None,
                        "summary": str((directory / SUMMARY).relative_to(root)) if summary else None})
    prefix = args.output_prefix
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", prefix):
        raise ValueError("--output-prefix must use letters, digits, underscores, or hyphens")
    atomic_json(root / f"{prefix}.json", {"schema_version": 1, "generated_at": now(), "runs": entries})
    fixed = ["run_dir", "label", "status", "started_at", "finished_at", "exit_code", "model", "url",
             "aiperf_version", "aiperf_schema_version", "server_job_id", "errors"]
    fields = fixed + sorted({key for row in rows for key in row} - set(fixed))
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=root, delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(f.name)
    os.replace(temporary, root / f"{prefix}.csv")
    print(f"Indexed {len(rows)} runs: {root / (prefix + '.json')} and {root / (prefix + '.csv')}")
    for row in rows:
        if row["errors"]: print(f"Warning: {row['run_dir']}: {row['errors']}", file=sys.stderr)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    model, tokenizer = model_defaults()
    run = commands.add_parser("benchmark", help="Benchmark an already-running OpenAI-compatible vLLM server")
    run.add_argument("--url", default="http://localhost:8040", help="Server base URL; trailing /v1 is accepted")
    run.add_argument("--endpoint", default="/v1/chat/completions")
    run.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", model), help="Served model name; defaults to models/models.txt primary name")
    run.add_argument("--tokenizer", default=tokenizer, help="Matching tokenizer path/HF ID; defaults to primary model")
    run.add_argument("--concurrency", type=positive, default=1)
    run.add_argument("--requests", type=positive, default=30)
    run.add_argument("--warmup", type=nonnegative, default=3)
    run.add_argument("--input-tokens", type=positive, default=512)
    run.add_argument("--output-tokens", type=positive, default=256)
    run.add_argument("--seed", type=nonnegative, default=42)
    run.add_argument("--timeout", type=positive_float, default=600)
    run.add_argument("--request-rate", type=positive_float)
    run.add_argument("--allow-eos", action="store_true", help="Allow early EOS; default requests fixed-length output via vLLM ignore_eos")
    run.add_argument("--label", default="benchmark")
    run.add_argument("--server-job-id", default="", help="Serving Slurm job ID for reference; does not start or query a job")
    for command in (run, commands.add_parser("report", help="Scan result directories into an index JSON and CSV")):
        command.add_argument("--root", type=Path, default=ROOT / "aiperf-out", help="AIPerf output root")
        if command is not run:
            command.add_argument("--output-prefix", default="index")
    args = parser.parse_args()
    try:
        return benchmark(args) if args.action == "benchmark" else report(args)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"AIPerf tools: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
