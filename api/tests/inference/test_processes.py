import unittest
from unittest.mock import patch
from api.inference.processes import visible_cuda_device


class ProcessTests(unittest.TestCase):
    def test_cuda_remapping_survives_child_isolation(self):
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '3,GPU-fixture'}):
            self.assertEqual(visible_cuda_device(0), '3')
            self.assertEqual(visible_cuda_device(1), 'GPU-fixture')
        with patch.dict('os.environ', {}, clear=True):
            self.assertEqual(visible_cuda_device(2), '2')
