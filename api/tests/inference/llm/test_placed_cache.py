import unittest
from unittest.mock import patch
import torch
from api.inference.llm.offload import ExpertBank, ExpertCache
from api.inference.llm.placed_cache import PlacedExpertCache
from api.inference.resources import ResourceManager


class PlacedCacheTests(unittest.TestCase):
    def test_virtual_devices_copy_full_bank_once_and_hold_remote_workspace(self):
        resources=ResourceManager(1000,{0:100,1:100})
        bank=ExpertBank({'a':{'weight':torch.arange(15,dtype=torch.float32)},
                         'b':{'weight':torch.arange(15,dtype=torch.float32)+10}})
        def cpu_cache(bank, capacity, device, **kwargs):
            return ExpertCache(bank,capacity,'cpu',resource_device=int(device[5:]),**kwargs)
        with patch('api.inference.llm.placed_cache.GIB',16), patch('api.inference.llm.placed_cache.REMOTE_SCRATCH',16), \
             patch('api.inference.llm.placed_cache.ExpertCache',side_effect=cpu_cache), \
             patch('api.inference.llm.placed_cache.torch.cuda.synchronize'):
            cache=PlacedExpertCache(bank,100,'cuda:0',resources=resources,owner='test')
            with cache.execution():
                self.assertTrue(cache.plan.fully_resident)
                self.assertEqual(set(cache.plan.assignments.values()),{0,1})
                self.assertEqual(cache.resident_bytes,120)
                self.assertEqual(cache.misses,2)
                for key in ('a','b'):
                    with cache.use(key) as tensors:
                        torch.testing.assert_close(tensors['weight'],bank.get(key)['weight'])
                self.assertTrue(any('scratch' in key for key in resources.snapshot()['reservations']))
            self.assertFalse(any('scratch' in key for key in resources.snapshot()['reservations']))
            with cache.execution():
                with cache.use('a'): pass
            self.assertEqual(cache.misses,2)
            cache.close()
            self.assertEqual(resources.snapshot()['reservations'],{})
