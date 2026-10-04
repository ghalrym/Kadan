import threading
import unittest

import torch

from api.inference.offload import ExpertBank, ExpertCache


class OffloadTests(unittest.TestCase):
    def bank(self):
        return ExpertBank({key: {'w': torch.full((4,), key, dtype=torch.uint8)} for key in range(3)})

    def test_cpu_bank_shares_storage_not_full_clone(self):
        whole = torch.arange(12, dtype=torch.uint8)
        bank = ExpertBank({0: {'w': whole[:4]}, 1: {'w': whole[4:8]}})
        self.assertEqual(bank.host_bytes, 12)
        self.assertEqual(bank.max_expert_bytes, 4)
        self.assertEqual(bank.get(0)['w'].data_ptr(), whole.data_ptr())

    def test_cache_hits_lru_eviction_and_bytes(self):
        cache = ExpertCache(self.bank(), 8, 'cpu')
        for key in [0, 1, 0, 2]:
            with cache.use(key) as expert:
                self.assertTrue(torch.equal(expert['w'], torch.full((4,), key, dtype=torch.uint8)))
                self.assertLessEqual(cache.resident_bytes, 8)
        self.assertEqual((cache.hits, cache.misses, cache.evictions), (1, 3, 1))
        with self.assertRaises(RuntimeError):
            expert['w']  # A caller's loop variable cannot retain evicted storage.
        with cache.use(1):
            pass
        self.assertEqual(cache.misses, 4)
        cache.close()
        self.assertEqual(cache.resident_bytes, 0)
        with self.assertRaises(RuntimeError):
            with cache.use(0):
                pass

    def test_rejection_and_exception_release(self):
        cache = ExpertCache(self.bank(), 3, 'cpu')
        with self.assertRaises(MemoryError):
            with cache.use(0):
                pass
        self.assertEqual(cache.resident_bytes, 0)
        cache = ExpertCache(self.bank(), 4, 'cpu')
        with self.assertRaisesRegex(ValueError, 'failure'):
            with cache.use(0):
                with self.assertRaises(RuntimeError):
                    cache.clear()
                with self.assertRaises(RuntimeError):
                    with cache.use(1):
                        pass
                raise ValueError('failure')
        with cache.use(1):
            pass
        self.assertEqual(cache.resident_bytes, 4)

    def test_concurrent_use_waits_until_lease_ends(self):
        cache = ExpertCache(self.bank(), 4, 'cpu')
        started, finished = threading.Event(), threading.Event()
        def second():
            started.set()
            with cache.use(1):
                finished.set()
        with cache.use(0):
            thread = threading.Thread(target=second)
            thread.start()
            self.assertTrue(started.wait(1))
            self.assertFalse(finished.is_set())
        thread.join(2)
        self.assertTrue(finished.is_set())
        self.assertEqual(cache.resident_bytes, 4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA validation requires target GPU')
    def test_cuda_transfer_compute_eviction_and_close(self):
        cache = ExpertCache(self.bank(), 4, 'cuda:0')
        outputs = []
        for key in [0, 1, 2, 0]:
            with cache.use(key) as expert:
                outputs.append(expert['w'].float().sum())
                self.assertEqual(cache.resident_bytes, 4)
        cache.close()
        self.assertEqual([value.item() for value in outputs], [0., 4., 8., 0.])
        self.assertEqual(cache.resident_bytes, 0)


class SharedBudgetOffloadTests(unittest.TestCase):
    def setup_cache(self, capacity=12):
        from api.inference.resources import ResourceManager
        bank = ExpertBank({key: {'w': torch.full((4,), key, dtype=torch.uint8)} for key in range(3)})
        resources = ResourceManager(100, {0: capacity})
        cache = ExpertCache(bank, capacity, 'cpu', resources=resources,
                            owner='test-model', resource_device=0)
        return bank, resources, cache

    def test_kv_growth_frees_real_cached_storage_but_keeps_host_bank(self):
        import weakref
        bank, resources, cache = self.setup_cache()
        host_refs = [weakref.ref(bank.get(key)['w']) for key in range(3)]
        cached_refs = []
        for key in range(3):
            with cache.use(key) as tensors:
                cached_refs.append(weakref.ref(tensors['w']))
        self.assertEqual(cache.resident_bytes, 12)
        self.assertEqual(len(resources.snapshot()['reservations']), 3)
        kv = resources.reserve('long-request-kv', 'llm', device_bytes={0: 8})
        with kv.lease():
            self.assertEqual(cache.resident_bytes, 4)
            self.assertEqual(sum(ref() is None for ref in cached_refs), 2)
            self.assertTrue(all(ref() is not None for ref in host_refs))
            # Re-demanding an evicted expert replaces the remaining cached one;
            # request KV owns eight bytes, leaving room for exactly one expert.
            with cache.use(0) as tensors:
                torch.testing.assert_close(tensors['w'], bank.get(0)['w'])
            self.assertEqual(cache.resident_bytes, 4)
        kv.release()
        for key in range(3):
            with cache.use(key):
                pass
        self.assertEqual(cache.resident_bytes, 12)
        cache.close()
        self.assertEqual(resources.snapshot()['reservations'], {})
        self.assertTrue(all(ref() is not None for ref in host_refs))

    def test_active_expert_is_protected_from_request_admission(self):
        from api.inference.resources import ResourceExhausted
        _, resources, cache = self.setup_cache(4)
        with cache.use(0) as tensors:
            states = resources.snapshot()['reservations'].values()
            self.assertEqual(sum(state['active_leases'] for state in states), 1)
            with self.assertRaises(ResourceExhausted):
                resources.reserve('kv', 'llm', device_bytes={0: 1})
            self.assertEqual(tensors['w'].numel(), 4)
            self.assertEqual(cache.resident_bytes, 4)
        kv = resources.reserve('kv', 'llm', device_bytes={0: 4})
        self.assertEqual(cache.resident_bytes, 0)
        kv.release()

    def test_nonblocking_eviction_avoids_cache_admission_lock_inversion(self):
        from api.inference.resources import ResourceExhausted
        _, resources, cache = self.setup_cache(4)
        with cache.use(0):
            pass
        locked, finish = threading.Event(), threading.Event()
        def hold_cache_lock():
            with cache._lock:
                locked.set()
                finish.wait(2)
        holder = threading.Thread(target=hold_cache_lock)
        holder.start()
        self.assertTrue(locked.wait(1))
        try:
            with self.assertRaises(ResourceExhausted):
                resources.reserve('kv', 'llm', device_bytes={0: 4})
        finally:
            finish.set()
            holder.join(2)
        self.assertFalse(holder.is_alive())
        kv = resources.reserve('kv', 'llm', device_bytes={0: 4})
        self.assertEqual(cache.resident_bytes, 0)
        kv.release()

    def test_partial_copy_failure_frees_tensors_before_releasing_accounting(self):
        from unittest.mock import patch
        import weakref
        _, resources, cache = self.setup_cache()
        refs = []
        def broken_copy(source, destination):
            destination['w'] = source['w'].clone()
            refs.append(weakref.ref(destination['w']))
            raise RuntimeError('copy failed')
        with patch.object(cache, '_copy', broken_copy):
            with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                with cache.use(0):
                    pass
        self.assertIsNone(refs[0]())
        self.assertEqual(cache.resident_bytes, 0)
        self.assertEqual(resources.snapshot()['reservations'], {})

    def test_failed_admission_does_not_start_copy(self):
        from unittest.mock import patch
        from api.inference.resources import ResourceExhausted
        _, resources, cache = self.setup_cache(4)
        kv = resources.reserve('kv', 'llm', device_bytes={0: 4})
        with kv.lease(), patch.object(cache, '_copy') as copy:
            with self.assertRaises(ResourceExhausted):
                with cache.use(0):
                    pass
            copy.assert_not_called()
        self.assertEqual(cache.resident_bytes, 0)
        kv.release()

    def test_transfer_is_protected_before_caller_gets_lease(self):
        from unittest.mock import patch
        from api.inference.resources import ResourceExhausted
        _, resources, cache = self.setup_cache(4)
        original_copy = cache._copy
        def inspect_copy(source, destination):
            self.assertEqual(sum(state['active_leases'] for state in resources.snapshot()['reservations'].values()), 1)
            with self.assertRaises(ResourceExhausted):
                resources.reserve('racing-kv', 'llm', device_bytes={0: 4})
            original_copy(source, destination)
        with patch.object(cache, '_copy', inspect_copy):
            with cache.use(0):
                pass
        cache.close()
        self.assertEqual(resources.snapshot()['reservations'], {})
