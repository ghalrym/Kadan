import unittest

from api.tests.services.whisper_contract import assert_checkpoint_contract


class CheckpointTests(unittest.TestCase):
    def test_native_checkpoint_contract(self):
        assert_checkpoint_contract(self, 'small',
            '9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794')
