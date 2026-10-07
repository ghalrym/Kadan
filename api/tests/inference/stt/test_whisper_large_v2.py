import unittest

from api.tests.inference.stt.whisper_contract import assert_checkpoint_contract


class CheckpointTests(unittest.TestCase):
    def test_native_checkpoint_contract(self):
        assert_checkpoint_contract(self, 'large-v2',
            '81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524')
