import unittest

from api.tests.inference.stt.whisper_contract import assert_checkpoint_contract


class CheckpointTests(unittest.TestCase):
    def test_inference_checkpoint_contract(self):
        assert_checkpoint_contract(self, 'tiny',
            '65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9')
