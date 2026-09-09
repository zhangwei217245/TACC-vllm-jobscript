import contextlib
import io
import os
import tempfile
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

    @patch.dict('os.environ', {'HF_TOKEN': 'never-report-this', 'MASTER_PORT': '29500'})
    @patch.object(collector, 'run')
    def test_probe_without_utilities(self, run):
        run.return_value = {'status': 'unavailable'}
        result = collector.probe({'command_timeout': 1, 'paths': ['/nonexistent/dgx-model']})
        self.assertEqual(result['gpus'], [])
        self.assertNotIn('HF_TOKEN', result['environment'])
        self.assertEqual(result['environment']['MASTER_PORT'], '29500')
        self.assertIn('error', result['storage']['/nonexistent/dgx-model'])

    @patch.object(collector, 'run')
    def test_default_cli_writes_local_report(self, run):
        run.return_value = {'status': 'unavailable'}
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch.object(sys, 'argv', ['collect_dgx_info.py']), \
                     patch.object(collector.socket, 'gethostname', return_value='dgx01'), \
                     contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(collector.main(), 0)
                outputs = list(Path('.').glob('dgx-inventory-dgx01-*.json'))
                self.assertEqual(len(outputs), 1)
                report = json.loads(outputs[0].read_text())
                self.assertEqual(report['inventory']['hostname'], 'dgx01')
                self.assertEqual(report['schema_version'], 2)
                self.assertIn('No GPUs', report['warnings'][0])
                self.assertIn('Host: dgx01', Path(str(outputs[0]) + '.txt').read_text())
                self.assertFalse(any(call.args[0][0] in ('ssh', 'srun', 'scontrol')
                                     for call in run.call_args_list))
            finally:
                os.chdir(previous)

    @patch.object(collector, 'run')
    def test_output_write_failure(self, run):
        run.return_value = {'status': 'unavailable'}
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, 'argv', ['collect_dgx_info.py', '--output', directory]), \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(collector.main(), 1)


if __name__ == '__main__':
    unittest.main()
