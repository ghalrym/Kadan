import unittest

from api.tests.inference.stt.whisper_contract import assert_checkpoint_contract


class CheckpointTests(unittest.TestCase):
    def test_native_checkpoint_contract(self):
        assert_checkpoint_contract(self, 'large-v3-turbo',
            'aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a')
