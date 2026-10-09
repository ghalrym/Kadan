import threading
import unittest
from api.inference.resources import (
    MemoryCapacity, ResourceBusy, ResourceCancelled, ResourceExhausted, ResourceManager,
)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.manager = ResourceManager(448, {0: 24, 1: 24})

    def test_unpressured_admission_does_not_scan_existing_residents(self):
        class NoScan(dict):
            def values(self):
                raise AssertionError('Unpressured admission scanned all residents')
            def items(self):
                raise AssertionError('Unpressured admission built eviction candidates')
        probes = []
        manager = ResourceManager(10000, {0: 10000},
            probe=lambda: (probes.append(1) or MemoryCapacity(10000, {0: 10000})))
        manager._residents = NoScan()
        handles = [manager.reserve(str(i), 'llm', 1, {0: 1}) for i in range(1000)]
        self.assertEqual(len(probes), 1000)
        self.assertEqual(manager._used_host, 1000)
        for handle in handles:
            handle.release()
            handle.release()
        self.assertEqual(manager._used_host, 0)
        self.assertEqual(manager._used_devices[0], 0)

    def test_eviction_callback_release_updates_totals_once(self):
        manager = ResourceManager(100, {0: 100})
        resident = manager.reserve('old', 'llm', 80, {0: 80}, lambda: resident.release())
        replacement = manager.reserve('new', 'speech', 90, {0: 90})
        resident.release()
        self.assertEqual(manager._used_host, 90)
        self.assertEqual(manager._used_devices[0], 90)
        replacement.release()
        self.assertEqual(manager._used_host, 0)

    def test_gpu_budgets_never_pool(self):
        with self.assertRaises(ResourceExhausted):
            self.manager.reserve('llm', 'llm', device_bytes={0: 25})
        reservation = self.manager.reserve('split', 'llm', device_bytes={0: 20, 1: 20})
        self.assertEqual(self.manager.snapshot()['reservations']['split']['device_bytes'], {0: 20, 1: 20})
        reservation.release()

    def test_active_work_cannot_be_evicted_and_exception_releases_lease(self):
        events = []
        llm = self.manager.reserve('llm', 'llm', 400, {0: 20}, lambda: events.append('unloaded'))
        with self.assertRaisesRegex(RuntimeError, 'generation failed'):
            with llm.lease():
                with self.assertRaises(ResourceExhausted):
                    self.manager.reserve('image', 'image', 100, {0: 20})
                with self.assertRaises(ResourceBusy):
                    llm.release()
                raise RuntimeError('generation failed')
        self.assertEqual(events, [])
        image = self.manager.reserve('image', 'image', 100, {0: 20})
        self.assertEqual(events, ['unloaded'])
        with self.assertRaises(ResourceBusy):
            with llm.lease(): pass
        image.release()
        self.assertEqual(self.manager.snapshot()['reservations'], {})

    def test_exclusive_future_modality_frees_all_llm_gpu_ownership(self):
        events = []
        llm = self.manager.reserve('llm', 'llm', 100, {0: 10}, lambda: events.append('freed'))
        with self.manager.exclusive('video'):
            self.assertEqual(events, ['freed'])
            video = self.manager.reserve('video', 'video', 100, {0: 24})
            with video.lease():
                with self.assertRaises(ResourceBusy):
                    self.manager.reserve('speech', 'speech', 1, {1: 1})
            video.release()
        llm.release()  # stale handle is harmless
        self.assertIsNone(self.manager.snapshot()['exclusive_owner'])

    def test_eviction_failure_keeps_accounting_and_clears_exclusive(self):
        def broken(): raise OSError('device did not release')
        self.manager.reserve('llm', 'llm', 100, {0: 20}, broken)
        with self.assertRaises(OSError):
            with self.manager.exclusive('image'): pass
        snapshot = self.manager.snapshot()
        self.assertIsNone(snapshot['exclusive_owner'])
        self.assertFalse(snapshot['reservations']['llm']['evicting'])
        self.assertEqual(snapshot['reservations']['llm']['device_bytes'], {0: 20})

    def test_cancellation_before_admission_and_during_eviction(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(ResourceCancelled):
            self.manager.reserve('llm', 'llm', cancel_event=cancel)
        cancel.clear()
        self.manager.reserve('llm', 'llm', 100, {0: 20}, cancel.set)
        with self.assertRaises(ResourceCancelled):
            self.manager.reserve('image', 'image', 100, {0: 20}, cancel_event=cancel)
        self.assertEqual(self.manager.snapshot()['reservations'], {})

    def test_physical_probe_rejects_external_memory_pressure(self):
        manager = ResourceManager(448, {0: 24}, probe=lambda: MemoryCapacity(100, {0: 2}))
        with self.assertRaises(ResourceExhausted):
            manager.reserve('llm', 'llm', 80, {0: 3})
        self.assertEqual(manager.snapshot()['reservations'], {})

    def test_stale_handle_cannot_release_new_owner(self):
        old = self.manager.reserve('llm', 'llm', 1, {0: 1})
        old.release()
        new = self.manager.reserve('llm', 'llm', 1, {0: 1})
        old.release()
        with new.lease():
            self.assertEqual(self.manager.snapshot()['reservations']['llm']['active_leases'], 1)

    def test_eviction_callback_can_release_and_read_snapshot(self):
        handle = None
        def cleanup():
            self.manager.snapshot()
            handle.release()
        handle = self.manager.reserve('llm', 'llm', 400, {0: 24}, cleanup)
        self.manager.reserve('image', 'image', 100, {0: 24})
        self.assertNotIn('llm', self.manager.snapshot()['reservations'])

    def test_concurrent_reservations_never_overcommit(self):
        start = threading.Barrier(3)
        results = []
        def reserve(owner):
            start.wait()
            try:
                self.manager.reserve(owner, 'decision', 300, {0: 20})
                results.append('ok')
            except ResourceExhausted:
                results.append('full')
        workers = [threading.Thread(target=reserve, args=(str(index),)) for index in range(2)]
        for worker in workers: worker.start()
        start.wait()
        for worker in workers: worker.join(2)
        self.assertCountEqual(results, ['ok', 'full'])

    def test_gpu_handoff_preserves_host_backing(self):
        host = self.manager.reserve('llm:host', 'llm', host_bytes=300)
        self.manager.reserve('llm:gpu', 'llm', device_bytes={0: 20}, evict=lambda: None)
        with self.manager.exclusive('image'):
            snapshot = self.manager.snapshot()['reservations']
            self.assertIn('llm:host', snapshot)
            self.assertNotIn('llm:gpu', snapshot)
        host.release()

    def test_waiting_admission_can_be_cancelled(self):
        entered, release, cancel = threading.Event(), threading.Event(), threading.Event()
        def cleanup():
            entered.set()
            release.wait(3)
        self.manager.reserve('llm', 'llm', 400, {0: 20}, cleanup)
        winner = threading.Thread(target=lambda: self.manager.reserve('image', 'image', 100, {0: 20}))
        winner.start()
        self.assertTrue(entered.wait(1))
        outcome = []
        def waiter():
            try:
                self.manager.reserve('speech', 'speech', cancel_event=cancel)
            except ResourceCancelled:
                outcome.append('cancelled')
        waiting = threading.Thread(target=waiter)
        waiting.start()
        cancel.set()
        waiting.join(1)
        release.set()
        winner.join(1)
        self.assertEqual(outcome, ['cancelled'])

    def test_invalid_and_unknown_device_budgets(self):
        for host in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                self.manager.reserve('invalid', 'llm', host)
        with self.assertRaises(ResourceExhausted):
            self.manager.reserve('missing-gpu', 'llm', device_bytes={2: 1})


class CgroupProbeTests(unittest.TestCase):
    def probe(self, files):
        from unittest.mock import patch
        from api.inference.resources import probe_memory
        def read(path):
            try:
                return files[str(path)]
            except KeyError:
                raise FileNotFoundError(str(path)) from None
        with patch('api.inference.resources._read_text', side_effect=read), patch.dict('sys.modules', {'torch': None}):
            return probe_memory().host_bytes

    def test_v2_nested_ancestor_limits_and_usage(self):
        files = {'/proc/meminfo': 'MemAvailable: 1000 kB\n',
                 '/proc/self/cgroup': '0::/parent/child\n',
                 '/proc/self/mountinfo': '1 0 0:1 / /cg rw - cgroup2 cgroup rw\n',
                 '/cg/parent/child/memory.max': '500000', '/cg/parent/child/memory.current': '100000',
                 '/cg/parent/memory.max': '350000', '/cg/parent/memory.current': '200000',
                 '/cg/memory.max': 'max'}
        self.assertEqual(self.probe(files), 150000)
        files['/cg/parent/memory.current'] = '400000'
        self.assertEqual(self.probe(files), 0)

    def test_mount_root_namespace_and_escaped_mountpoint(self):
        files = {'/proc/meminfo': 'MemAvailable: 1000 kB\n',
                 '/proc/self/cgroup': '0::/container/leaf',
                 '/proc/self/mountinfo': '1 0 0:1 /container /cg\\040space rw - cgroup2 cgroup rw\n',
                 '/cg space/leaf/memory.max': '2000', '/cg space/leaf/memory.current': '1200'}
        self.assertEqual(self.probe(files), 800)
        files['/proc/self/cgroup'] = '0::/leaf'
        self.assertEqual(self.probe(files), 800)
        files['/proc/self/cgroup'] = '0::/'
        files['/cg space/memory.max'] = '1000'
        files['/cg space/memory.current'] = '300'
        self.assertEqual(self.probe(files), 700)

    def test_v1_memory_controller_and_unlimited_parent(self):
        files = {'/proc/meminfo': 'MemAvailable: 1000 kB\n',
                 '/proc/self/cgroup': '3:cpu:/wrong\n5:memory:/service\n',
                 '/proc/self/mountinfo': '1 0 0:1 / /mem rw - cgroup cgroup rw,memory\n',
                 '/mem/service/memory.limit_in_bytes': '6000', '/mem/service/memory.usage_in_bytes': '4500',
                 '/mem/memory.limit_in_bytes': '9223372036854771712', '/mem/memory.usage_in_bytes': '0'}
        self.assertEqual(self.probe(files), 1500)

    def test_host_minimum_unlimited_and_no_cgroup_fallback(self):
        files = {'/proc/meminfo': 'MemAvailable: 2 kB\n', '/proc/self/cgroup': '0::/',
                 '/proc/self/mountinfo': '1 0 0:1 / /cg rw - cgroup2 cgroup rw\n',
                 '/cg/memory.max': '100000', '/cg/memory.current': '1'}
        self.assertEqual(self.probe(files), 2048)
        files['/cg/memory.max'] = 'max'
        self.assertEqual(self.probe(files), 2048)
        del files['/proc/self/mountinfo']
        self.assertEqual(self.probe(files), 2048)
        files['/proc/meminfo'] = 'MemAvailable: 0 kB\n'
        self.assertEqual(self.probe(files), 0)

    def test_finite_limit_with_unreadable_usage_fails_closed(self):
        files = {'/proc/meminfo': 'MemAvailable: 1000 kB\n', '/proc/self/cgroup': '0::/',
                 '/proc/self/mountinfo': '1 0 0:1 / /cg rw - cgroup2 cgroup rw\n',
                 '/cg/memory.max': '1000'}
        self.assertEqual(self.probe(files), 0)


class PhysicalEvictionTests(unittest.TestCase):
    def test_physical_pressure_evicts_idle_resident_even_when_budget_fits(self):
        free = [100]
        manager = ResourceManager(1000, {0: 1000}, probe=lambda: MemoryCapacity(free[0], {0: free[0]}))
        events = []
        def cleanup():
            events.append('evict')
            free[0] = 100
        manager.reserve('llm', 'llm', 50, {0: 50}, cleanup)
        free[0] = 10
        manager.reserve('image', 'image', 50, {0: 50})
        self.assertEqual(events, ['evict'])
        self.assertEqual(set(manager.snapshot()['reservations']), {'image'})

    def test_physical_pressure_cannot_evict_active_or_admit_without_actual_release(self):
        free = [100]
        manager = ResourceManager(1000, {0: 1000}, probe=lambda: MemoryCapacity(free[0], {0: free[0]}))
        events = []
        resident = manager.reserve('llm', 'llm', 50, {0: 50}, lambda: events.append('evict'))
        free[0] = 10
        with resident.lease():
            with self.assertRaises(ResourceExhausted):
                manager.reserve('image', 'image', 50, {0: 50})
        self.assertEqual(events, [])
        with self.assertRaises(ResourceExhausted):
            manager.reserve('image', 'image', 50, {0: 50})
        self.assertEqual(events, ['evict'])
        self.assertNotIn('image', manager.snapshot()['reservations'])

    def test_busy_candidate_is_skipped_for_alterinference_idle_resident(self):
        free = [100]
        events = []
        manager = ResourceManager(1000, {0: 1000}, probe=lambda: MemoryCapacity(1000, {0: free[0]}))
        def busy():
            events.append('busy')
            raise ResourceBusy('construction in progress')
        def cleanup():
            events.append('clean')
            free[0] = 100
        manager.reserve('busy', 'llm', device_bytes={0: 20}, evict=busy)
        manager.reserve('idle', 'llm', device_bytes={0: 20}, evict=cleanup)
        free[0] = 10
        manager.reserve('image', 'image', device_bytes={0: 50})
        self.assertEqual(events, ['busy', 'clean'])
        self.assertIn('busy', manager.snapshot()['reservations'])

    def test_gpu_pressure_does_not_evict_unrelated_host_reservation(self):
        free = [100]
        events = []
        manager = ResourceManager(1000, {0: 1000}, probe=lambda: MemoryCapacity(1000, {0: free[0]}))
        manager.reserve('host', 'llm', host_bytes=100, evict=lambda: events.append('host'))
        def cleanup():
            events.append('device')
            free[0] = 100
        manager.reserve('device', 'llm', device_bytes={0: 20}, evict=cleanup)
        free[0] = 10
        manager.reserve('image', 'image', device_bytes={0: 50})
        self.assertEqual(events, ['device'])
