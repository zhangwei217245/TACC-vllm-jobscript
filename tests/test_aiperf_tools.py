"""Exercise the benchmark/report contract without installing AIPerf or using GPUs."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tacc benchmark ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "aiperf-out"
        mock = self.root / "mock-aiperf"
        mock.write_text("#!" + sys.executable + "\n" + '''
import json, os, sys
from pathlib import Path
if '--version' in sys.argv:
    print('AIPerf 0.13.0'); sys.exit(0)
args = sys.argv[1:]
out = Path(args[args.index('--output-artifact-dir') + 1])
if '--api-key' in args:
    assert args[args.index('--api-key') + 1] == '${AIPERF_BENCH_API_KEY}'
    assert os.environ['AIPERF_BENCH_API_KEY'] == 'secret-for-test'
if os.environ.get('MOCK_FAILURE'):
    print('simulated failure'); sys.exit(7)
if not os.environ.get('MOCK_NO_SUMMARY'):
    (out / 'profile_export_aiperf.json').write_text(json.dumps({
        'schema_version': '1.0',
        'time_to_first_token': {'unit': 'ms', 'avg': 120.5, 'p99': 200},
        'output_token_throughput': {'unit': 'tokens/sec', 'avg': 42},
        'inter_token_latency': {'unit': 'ms', 'avg': float('nan')},
    }))
''')
        mock.chmod(0o755)
        self.env = dict(os.environ, AIPERF_BIN=str(mock), AIPERF_PYTHON=sys.executable,
                        VLLM_API_KEY="secret-for-test")

    def run_script(self, script, *args, **env):
        return subprocess.run(["bash", str(ROOT / "utility" / script),
                               "--root", str(self.output), *args],
                              env={**self.env, **env}, text=True, capture_output=True)

    def benchmark(self, **env):
        return self.run_script("benchmark_aiperf.sh", "--model", "served-model",
                               "--tokenizer", "/path with spaces/tokenizer", "--url",
                               "http://localhost:8040/v1/", "--label", "tp2 pp2", **env)

    def test_success_and_index(self):
        for _ in range(2):
            result = self.benchmark()
            self.assertEqual(result.returncode, 0, result.stderr)
        manifests = list(self.output.glob("*/run.json"))
        self.assertEqual(len(manifests), 2)
        for path in manifests:
            text = path.read_text()
            self.assertNotIn("secret-for-test", text)
            meta = json.loads(text)
            self.assertEqual(meta["status"], "completed")
            self.assertEqual(meta["url"], "http://localhost:8040")
            self.assertEqual(meta["tokenizer"], "/path with spaces/tokenizer")
            extras = meta["command"][meta["command"].index("--extra-inputs") + 1]
            self.assertEqual(json.loads(extras), {"temperature": 0, "ignore_eos": True})
        self.assertFalse((self.output / "index.csv").exists())
        result = self.run_script("report_aiperf.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        with (self.output / "index.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["time_to_first_token.unit"], "ms")
        self.assertEqual(rows[0]["time_to_first_token.p99"], "200")
        self.assertNotIn("inter_token_latency.avg", rows[0])
        index = json.loads((self.output / "index.json").read_text())
        self.assertEqual(len(index["runs"]), 2)
        self.assertTrue((self.output / index["runs"][0]["summary"]).is_file())

    def test_failure_missing_export_and_malformed_report(self):
        self.assertEqual(self.benchmark(MOCK_FAILURE="1").returncode, 7)
        self.assertEqual(self.benchmark(MOCK_NO_SUMMARY="1").returncode, 2)
        for manifest in self.output.glob("*/run.json"):
            self.assertEqual(json.loads(manifest.read_text())["status"], "failed")
        broken = self.output / "broken"
        broken.mkdir()
        (broken / "run.json").write_text("{broken")
        imported = self.output / "native"
        imported.mkdir()
        (imported / "profile_export_aiperf.json").write_text('{"metric":{"unit":"ms","avg":1}}')
        result = self.run_script("report_aiperf.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = json.loads((self.output / "index.json").read_text())["runs"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(next(r for r in rows if r["run_dir"] == "native")["status"], "imported")
        self.assertIn("Invalid run.json", next(r for r in rows if r["run_dir"] == "broken")["errors"])

    def test_empty_report_and_validation(self):
        self.output.mkdir()
        self.assertEqual(self.run_script("report_aiperf.sh").returncode, 0)
        self.assertEqual(json.loads((self.output / "index.json").read_text())["runs"], [])
        result = self.run_script("benchmark_aiperf.sh", "--requests", "1", "--concurrency", "2")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(list(self.output.glob("*/run.json")))

    def test_submit_creates_log_directory_before_sbatch(self):
        fake = self.root / "sbatch"
        fake.write_text("#!" + sys.executable + "\n" + '''
import json, os, sys
from pathlib import Path
assert Path(os.environ['LOG_DIR']).is_dir()
print(json.dumps(sys.argv[1:]))
''')
        fake.chmod(0o755)
        env = dict(self.env, PATH=str(self.root) + os.pathsep + os.environ["PATH"],
                   LOG_DIR=str(self.root / "logs with spaces"), DEPLOY_KIT_ROOT=str(ROOT))
        for opts in ([], ["--time=00:10:00", "--dry-run"]):
            result = subprocess.run(["bash", str(ROOT / "dgxspark/submit-vllm.sh"), *opts],
                                    cwd=self.root, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            args = json.loads(result.stdout)
            self.assertIn("--output=" + env["LOG_DIR"] + "/vllm-run-%j.out", args)
            self.assertIn(str(ROOT / "dgxspark/slurm-vllm.sbatch"), args)
            if opts:
                self.assertEqual(args[-1], "--dry-run")
                self.assertIn("--time=00:10:00", args)


if __name__ == "__main__":
    unittest.main()
