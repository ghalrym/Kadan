import unittest

from api.tests.inference.stt.whisper_contract import assert_checkpoint_contract


class CheckpointTests(unittest.TestCase):
    def test_inference_checkpoint_contract(self):
        assert_checkpoint_contract(self, 'tiny.en',
            'd3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03')
