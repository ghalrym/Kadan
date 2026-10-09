import unittest

from api.tests.inference.stt.whisper_contract import assert_checkpoint_contract


class CheckpointTests(unittest.TestCase):
    def test_inference_checkpoint_contract(self):
        assert_checkpoint_contract(self, 'medium',
            '345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1')
