import gc
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from api.inference.checkpoint import SafeTensorReader


class SafeTensorReaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.tensors = {f'weight_{i}': torch.arange(8).reshape(2, 4) + i for i in range(32)}
        save_file(self.tensors, self.path / 'first.safetensors')
        save_file({'other': torch.ones(4)}, self.path / 'second.safetensors')
        weights = {name: 'first.safetensors' for name in self.tensors}
        weights['other'] = 'second.safetensors'
        (self.path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weights}))
        self.reader = SafeTensorReader(self.path)
        self.addCleanup(self.reader.close)

    def test_repeated_tensor_and_expert_reads_share_backing_memory(self):
        first = self.reader.tensor('weight_0')
        second = self.reader.tensor('weight_0')
        expert = self.reader.expert('weight_0', 0)
        self.assertEqual(first.data_ptr(), second.data_ptr())
        self.assertEqual(first.data_ptr(), expert.data_ptr())
        torch.testing.assert_close(expert, self.tensors['weight_0'][0])

    @unittest.skipUnless(Path('/proc/self/maps').exists(), 'Linux mapping accounting required')
    def test_retained_tensors_do_not_create_a_mapping_per_tensor(self):
        tensors = [self.reader.tensor(name) for name in self.tensors]
        mappings = [line for line in Path('/proc/self/maps').read_text().splitlines()
                    if str(self.path / 'first.safetensors') in line]
        self.assertGreater(len(mappings), 0)
        self.assertLessEqual(len(mappings), 2)
        self.reader.close()
        torch.testing.assert_close(tensors[-1], self.tensors['weight_31'])
        del tensors
        gc.collect()
        self.assertNotIn(str(self.path / 'first.safetensors'), Path('/proc/self/maps').read_text())

    def test_tensors_from_multiple_shards_survive_reader_cleanup(self):
        first = self.reader.tensor('weight_1')
        other = self.reader.tensor('other')
        self.reader.close()
        self.reader.close()
        torch.testing.assert_close(first, self.tensors['weight_1'])
        torch.testing.assert_close(other, torch.ones(4))

    def test_failed_lookup_does_not_prevent_cleanup(self):
        tensor = self.reader.tensor('weight_0')
        with self.assertRaises(KeyError):
            self.reader.tensor('missing')
        self.reader.close()
        torch.testing.assert_close(tensor, self.tensors['weight_0'])
