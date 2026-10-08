import unittest
from types import SimpleNamespace

import torch

from cached_only import CachedOnlyBlock, tensor_record


class Layer:
    def __init__(self):
        self.k = self.v = None

    def get(self):
        return self.k, self.v


class DispatchTests(unittest.TestCase):
    def make(self, retain_view=False, mutate=False):
        calls = []
        def forward(**kwargs):
            calls.append('eager:' + kwargs['kv_cache_mode'])
            if kwargs['kv_cache_mode'] == 'extract':
                layer = kwargs['layer_cache']
                k = torch.ones((1, 64, 32, 128), dtype=torch.bfloat16)[:, :27]
                layer.k = k if retain_view else k.clone()
                layer.v = k.clone()
            return None
        def compile_fn(eager):
            def compiled(**kwargs):
                calls.append('compiled:' + kwargs['kv_cache_mode'])
                if mutate:
                    kwargs['layer_cache'].k.add_(1)
                return None
            return compiled
        audit = CachedOnlyBlock(SimpleNamespace(forward=forward), 0, compile_fn, lambda *a, **k: None)
        audit.begin('D', True, 27)
        return audit, calls

    def extract(self, audit, layer):
        audit(layer_cache=layer, kv_cache_mode='extract', cache_write_slice=slice(0, 27))

    def test_extract_eager_cached_compiled_exact_storage(self):
        audit, calls = self.make()
        layer = Layer()
        self.extract(audit, layer)
        for _ in range(39):
            audit(layer_cache=layer, kv_cache_mode='cached')
        result = audit.finish()
        self.assertEqual(calls, ['eager:extract'] + ['compiled:cached'] * 39)
        self.assertEqual([r['storage_bytes'] for r in result['extracted']], [221184, 221184])

    def test_full_backing_view_is_rejected(self):
        audit, _ = self.make(retain_view=True)
        with self.assertRaises(AssertionError):
            self.extract(audit, Layer())

    def test_request_cache_identity_is_rejected(self):
        audit, _ = self.make()
        layer = Layer()
        self.extract(audit, layer)
        audit.begin('F', True, 27)
        layer.k = layer.v = None
        with self.assertRaisesRegex(AssertionError, 'reused'):
            self.extract(audit, layer)

    def test_new_request_cache_is_accepted(self):
        audit, _ = self.make()
        first = Layer()
        self.extract(audit, first)
        audit.begin('F', True, 27)
        second = Layer()
        self.extract(audit, second)
        self.assertIs(audit.current(), second)

    def test_cached_mutation_is_rejected(self):
        audit, _ = self.make(mutate=True)
        layer = Layer()
        self.extract(audit, layer)
        with self.assertRaises(AssertionError):
            audit(layer_cache=layer, kv_cache_mode='cached')

    def test_eager_baseline_never_enters_compiler(self):
        audit, calls = self.make()
        audit.begin('B', False, 27)
        layer = Layer()
        self.extract(audit, layer)
        audit(layer_cache=layer, kv_cache_mode='cached')
        self.assertEqual(calls, ['eager:extract', 'eager:cached'])


if __name__ == '__main__':
    unittest.main()
