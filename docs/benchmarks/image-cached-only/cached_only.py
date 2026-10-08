"""Experiment-only dispatch and cache audit, deliberately outside compiled graphs."""
import hashlib
import time
import weakref

import torch


def tensor_record(tensor, digest=False):
    storage = tensor.untyped_storage()
    expected = tensor.numel() * tensor.element_size()
    assert storage.nbytes() == expected, (storage.nbytes(), expected, tuple(tensor.shape))
    assert tensor.storage_offset() == 0, tensor.storage_offset()
    record = dict(shape=list(tensor.shape), bytes=expected, storage_bytes=storage.nbytes(),
                  pointer=storage.data_ptr(), offset=tensor.storage_offset(), version=tensor._version)
    if digest:
        record['sha256'] = hashlib.sha256(tensor.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    return record


class CachedOnlyBlock:
    """Call the original extraction eagerly; only cached forward enters torch.compile."""
    def __init__(self, block, index, compile_fn, emit):
        self.eager = block.forward
        self.compiled = compile_fn(self.eager)
        self.index = index
        self.emit = emit
        self.history = []
        self.phase = None
        self.current = None
        self.extracted = None
        self.extract_count = 0
        self.cached_count = 0
        self.audit_seconds = 0.0
        self.enabled = False
        self.expected_tokens = None

    def begin(self, phase, enabled, expected_tokens=None):
        if self.current is not None:
            self.history.append(self.current)
        self.current = None
        self.phase = phase
        self.enabled = enabled
        self.expected_tokens = expected_tokens
        self.extract_count = self.cached_count = 0
        self.audit_seconds = 0.0
        self.extracted = None

    def __call__(self, *args, **kwargs):
        mode = kwargs.get('kv_cache_mode')
        assert mode in ('extract', 'cached'), mode
        layer = kwargs['layer_cache']
        if mode == 'extract':
            assert self.extract_count == 0
            assert all(reference() is not layer for reference in self.history), 'Cache reused across requests'
            assert layer.k is None and layer.v is None, 'Extraction cache was not empty'
            self.current = weakref.ref(layer)
            # Keep this branch outside the compiled callable. Do not rely on an
            # in-graph clone/contiguous/detach to force compact storage.
            result = self.eager(*args, **kwargs)
            self.extract_count += 1
        else:
            assert self.extract_count == 1 and self.current() is layer
            result = (self.compiled if self.enabled else self.eager)(*args, **kwargs)
            self.cached_count += 1
        started = time.monotonic()
        detailed = mode == 'extract' or self.cached_count in (1, 39)
        records = [tensor_record(tensor, detailed) for tensor in layer.get()]
        assert records[0]['pointer'] != records[1]['pointer'], 'K/V share storage'
        if mode == 'extract':
            prefix = kwargs['cache_write_slice']
            tokens = prefix.stop - (prefix.start or 0)
            assert tokens == records[0]['shape'][1] == records[1]['shape'][1]
            if self.expected_tokens is not None:
                assert tokens == self.expected_tokens, (tokens, self.expected_tokens)
            self.extracted = records
        else:
            for initial, current in zip(self.extracted, records):
                for key in ('shape', 'bytes', 'storage_bytes', 'pointer', 'offset', 'version'):
                    assert initial[key] == current[key], (self.index, key, initial[key], current[key])
                if detailed:
                    assert initial['sha256'] == current['sha256'], 'Cached K/V values changed'
        self.audit_seconds += time.monotonic() - started
        if detailed:
            self.emit('cache.storage.audit', block=self.index, mode=mode,
                      cached_call=self.cached_count, records=records,
                      compiled=mode == 'cached' and self.enabled)
        return result

    def finish(self):
        assert self.extract_count == 1 and self.cached_count == 39, (self.extract_count, self.cached_count)
        return dict(block=self.index, audit_seconds=self.audit_seconds,
                    extracted=self.extracted, cached_calls=self.cached_count)
