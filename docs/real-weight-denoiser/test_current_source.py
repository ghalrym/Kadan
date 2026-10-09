"""Check the current source binding without changing historical numerical evidence."""
from pathlib import Path
import unittest

from bf16_contracts import ULYSSES_SOURCE
from trace_binding import sha256


class CurrentSourceTests(unittest.TestCase):
    def test_parallel_module_matches_current_execution_contract(self):
        source = Path(__file__).resolve().parents[2] / 'api/inference/image/sequence_parallel_attention.py'
        self.assertEqual(sha256(source), ULYSSES_SOURCE)
