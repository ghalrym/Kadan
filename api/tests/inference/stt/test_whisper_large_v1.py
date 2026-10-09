import unittest

from api.tests.inference.stt.whisper_contract import assert_checkpoint_contract


class CheckpointTests(unittest.TestCase):
    def test_inference_checkpoint_contract(self):
        assert_checkpoint_contract(self, 'large-v1',
            'e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a')
