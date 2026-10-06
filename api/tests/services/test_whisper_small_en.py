import unittest

from api.tests.services.whisper_contract import assert_checkpoint_contract


class CheckpointTests(unittest.TestCase):
    def test_native_checkpoint_contract(self):
        assert_checkpoint_contract(self, 'small.en',
            'f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872')
