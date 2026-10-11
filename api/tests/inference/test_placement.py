import unittest
from api.inference.placement import place_packed, select_device
from api.inference.resources import MemoryCapacity, ResourceExhausted, ResourceManager


class PlacementTests(unittest.TestCase):
    def test_one_gpu_preferred_when_complete_execution_fits(self):
        plan = place_packed({'a':40,'b':30},{0:100,1:100},0,{0:20,1:20})
        self.assertEqual(set(plan.assignments.values()),{0})
        self.assertTrue(plan.fully_resident)

    def test_split_only_when_required_and_headroom_is_not_weight_budget(self):
        plan = place_packed({'a':60,'b':50},{0:100,1:100},0,{0:30,1:30})
        self.assertEqual(set(plan.assignments.values()),{0,1})
        self.assertTrue(plan.fully_resident)
        spill = place_packed({'a':60,'b':60,'c':60},{0:100,1:100},0,{0:30,1:30})
        self.assertFalse(spill.fully_resident)
        self.assertEqual(spill.capacities,{0:70,1:70})
        with self.assertRaises(ResourceExhausted):
            place_packed({'a':80},{0:100,1:100},0,{0:30,1:30})

    def test_idle_residency_is_kept_until_admission_needs_space(self):
        resources = ResourceManager(1000,{0:100,1:100})
        evicted=[]
        first=resources.reserve('first','tts',device_bytes={0:80},evict=lambda:evicted.append('first'))
        self.assertEqual(select_device(resources,50),'cuda:1')
        self.assertEqual(evicted,[])
        second=resources.reserve('second','speech',device_bytes={1:50})
        self.assertEqual(select_device(resources,90),'cuda:0')
        self.assertEqual(evicted,[])
        third=resources.reserve('third','decision',device_bytes={0:90})
        self.assertEqual(evicted,['first'])
        self.assertIn('second',resources.snapshot()['reservations'])
        second.release();third.release();first.release()

    def test_physical_availability_and_active_leases_bound_reclaim(self):
        resources=ResourceManager(1000,{0:100,1:100},probe=lambda:MemoryCapacity(1000,{0:10,1:60}))
        self.assertEqual(select_device(resources,50),'cuda:1')
        self.assertEqual(select_device(resources,70),'cuda:0')
        self.assertEqual(select_device(resources,70,allow_cpu=True),'cuda:0')
        with self.assertRaises(ResourceExhausted):select_device(resources,101,'cuda:1')
        with self.assertRaises(ValueError):select_device(resources,1,'cuda:0,1')
        r=ResourceManager(1000,{0:100});held=r.reserve('held','llm',device_bytes={0:80},evict=lambda:None)
        with held.lease():
            self.assertEqual(r.available_devices(reclaim=True),{0:20})
        self.assertEqual(r.available_devices(reclaim=True),{0:100})

    def test_retained_model_prefers_existing_device_and_accounts_request_headroom(self):
        r=ResourceManager(1000,{0:100,1:100});r.reserve('tts','tts',device_bytes={1:80})
        self.assertEqual(select_device(r,90,retained=(1,80)),'cuda:1')
        self.assertEqual(select_device(r,101,retained=(1,80),allow_cpu=True),'cpu')
