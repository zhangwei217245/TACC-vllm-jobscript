import argparse
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_dgx_info as collector


class CollectorTests(unittest.TestCase):
    def test_missing_command(self):
        result = collector.run(['/nonexistent/dgx-test-command'])
        self.assertEqual(result['status'], 'unavailable')

    def test_timeout(self):
        result = collector.run([sys.executable, '-c', 'import time; time.sleep(10)'], timeout=0.05)
        self.assertEqual(result['status'], 'timeout')

    def test_partial_failure_report(self):
        nodes = [{'target': 'dgx1', 'status': 'failed'}]
        warnings = collector.summarize(nodes, 4)
        self.assertIn('collected 0', warnings[0])
        text = collector.render({'nodes': nodes, 'warnings': warnings, 'collected_at': 'now'})
        self.assertIn('dgx1: failed', text)

    @patch.dict('os.environ', {'HF_TOKEN': 'never-report-this', 'MASTER_PORT': '29500'})
    @patch.object(collector, 'run')
    def test_probe_without_utilities(self, run):
        run.return_value = {'status': 'unavailable'}
        result = collector.probe({'command_timeout': 1, 'paths': ['/nonexistent/dgx-model']})
        self.assertEqual(result['gpus'], [])
        self.assertNotIn('HF_TOKEN', result['environment'])
        self.assertEqual(result['environment']['MASTER_PORT'], '29500')
        self.assertIn('error', result['storage']['/nonexistent/dgx-model'])

    def test_streamed_source_executes(self):
        # Actually run the source through stdin, as SSH/Slurm will do.
        source = Path(collector.__file__).read_text().split('\nif __name__ ==')[0]
        source += '\nrun = lambda *a, **k: {"status": "unavailable"}\n'
        args = argparse.Namespace(transport='slurm', python=sys.executable, node_timeout=10)
        actual_run = collector.run

        def launch(argv, timeout, payload):
            self.assertIn('--nodelist=dgx1', argv)
            return actual_run([sys.executable, '-'], timeout, payload)

        with patch.object(collector, 'run', side_effect=launch):
            result = collector.collect_node('dgx1', args, source, {'command_timeout': 1})
        self.assertEqual(result['status'], 'collected', result)
        self.assertIn('hostname', result['inventory'])

    @patch.object(collector, 'run')
    def test_bad_remote_output(self, run):
        run.return_value = {'status': 'ok', 'stdout': 'login banner\nnot JSON', 'stderr': ''}
        args = argparse.Namespace(transport='ssh', python='python3', node_timeout=10)
        self.assertEqual(collector.collect_node('dgx1', args, '', {})['status'], 'failed')

    def test_cross_node_differences(self):
        def node(host, driver):
            return {'target': host, 'status': 'collected', 'inventory': {
                'hostname': host, 'architecture': 'x86_64', 'network': {'rdma': {'mlx5_0': {}}},
                'packages': {'vllm': 'example'}, 'gpus': [
                    {'name': 'Example GPU', 'memory.total': '80000', 'driver_version': driver}]}}
        findings = collector.summarize([node('dgx1', '1'), node('dgx2', '2')], 2)
        self.assertEqual(findings, ['GPU models/memory/driver differ across nodes.'])


if __name__ == '__main__':
    unittest.main()
