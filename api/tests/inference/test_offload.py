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
