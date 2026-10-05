"""Opt-in smoke of the actual isolated worker entrypoint, without loading a model."""
import os
from pathlib import Path
import subprocess
import unittest


class LTXImportTests(unittest.TestCase):
    @unittest.skipUnless(os.getenv('KADAN_LTX_SMOKE_PYTHON'), 'Isolated LTX dependency environment not configured')
    def test_actual_worker_subprocess_imports(self):
        worker = Path(__file__).parents[2] / 'inference/workers/ltx.py'
        result = subprocess.run([os.environ['KADAN_LTX_SMOKE_PYTHON'], str(worker), '--check-imports'],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'ltx-worker-imports-ok')
