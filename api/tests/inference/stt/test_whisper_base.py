import unittest

from api.tests.inference.stt.whisper_contract import assert_checkpoint_contract


class CheckpointTests(unittest.TestCase):
    def test_inference_checkpoint_contract(self):
        assert_checkpoint_contract(self, 'base',
            'ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e')
