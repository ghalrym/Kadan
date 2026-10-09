import unittest

from api.tests.inference.stt.whisper_contract import assert_checkpoint_contract


class CheckpointTests(unittest.TestCase):
    def test_inference_checkpoint_contract(self):
        assert_checkpoint_contract(self, 'medium.en',
            'd7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f')
