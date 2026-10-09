import unittest

from api.tests.inference.stt.whisper_contract import assert_checkpoint_contract


class CheckpointTests(unittest.TestCase):
    def test_inference_checkpoint_contract(self):
        assert_checkpoint_contract(self, 'large-v3',
            'e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb')
