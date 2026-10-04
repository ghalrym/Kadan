import threading
import unittest
from api.inference.resources import (
    MemoryCapacity, ResourceBusy, ResourceCancelled, ResourceExhausted, ResourceManager,
)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.manager = ResourceManager(448, {0: 24, 1: 24})

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
