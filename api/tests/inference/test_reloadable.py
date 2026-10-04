import gc
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
import weakref

import torch

from api.inference.reloadable import ReloadableAdapter
from api.inference.resources import ResourceBusy, ResourceCancelled, ResourceExhausted, ResourceManager


class ReloadableTests(unittest.TestCase):
    def setUp(self):
        self.resources = ResourceManager(128, {0: 64})
        self.created = 0
        self.references = []
        self.fail_build = False
        self.fail_close = False
        self.entered = None
        self.resume = None

    def factory(self, entry, path, resources, device, cancel_event):
        self.created += 1
        owner = f'fixture-{self.created}'
        host = resources.reserve(owner + ':host', 'llm', host_bytes=96)
        try:
            if self.fail_build:
                raise ValueError('controlled constructor failure')
            device_reservation = resources.reserve(owner + ':device', 'llm', device_bytes={0: 32})
        except BaseException:
            host.release()
            raise
        test = self
        class Inner:
            is_resident = True
            def __init__(self):
                self.payload = torch.ones(24, dtype=torch.float32)
            def generate(self, messages, max_new_tokens, cancel_event):
                with host.lease(cancel_event), device_reservation.lease(cancel_event):
                    if test.entered is not None:
                        test.entered.set()
                        test.resume.wait(2)
                    return f'answer-{owner}'
            def close(self):
                if test.fail_close:
                    raise RuntimeError('controlled close failure')
                self.payload = None
                self.is_resident = False
                device_reservation.release()
                host.release()
        inner = Inner()
        self.references.append(weakref.ref(inner.payload))
        return inner

    def build(self, factory=None):
        return ReloadableAdapter(factory or self.factory, SimpleNamespace(id='medium'),
                                 Path('/fixture'), self.resources, 'cuda:0')

    def test_idle_host_eviction_frees_tensor_and_reloads_next_generation(self):
        adapter = self.build()
        other = self.resources.reserve('image', 'image', host_bytes=96)
        gc.collect()
        self.assertIsNone(self.references[0]())
        self.assertFalse(adapter.is_resident)
        self.assertEqual(set(self.resources.snapshot()['reservations']), {'image'})
        other.release()
        self.assertEqual(adapter.generate([]), 'answer-fixture-2')
        self.assertTrue(adapter.is_resident)
        adapter.close()
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    def test_generation_protects_ownership_and_does_not_deadlock_eviction(self):
        adapter = self.build()
        self.entered, self.resume = threading.Event(), threading.Event()
        worker = threading.Thread(target=lambda: adapter.generate([]))
        worker.start()
        self.assertTrue(self.entered.wait(1))
        try:
            with self.assertRaises((ResourceBusy, ResourceExhausted)):
                self.resources.reserve('image', 'image', host_bytes=96)
            self.assertIsNotNone(self.references[0]())
        finally:
            self.resume.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        adapter.close()

    def test_same_thread_constructor_eviction_rejected_before_inner_exists(self):
        errors = []
        def factory(entry, path, resources, **kwargs):
            reservation = resources.reserve('constructing', 'llm', host_bytes=96)
            try:
                try:
                    resources.reserve('image', 'image', host_bytes=96)
                except (ResourceBusy, ResourceExhausted) as exc:
                    errors.append(exc)
            finally:
                reservation.release()
            return self.factory(entry, path, resources, **kwargs)
        adapter = self.build(factory)
        self.assertEqual(len(errors), 1)
        self.assertTrue(adapter.is_resident)
        adapter.close()

    def test_reload_failure_and_cancellation_do_not_leak_or_fake_success(self):
        adapter = self.build()
        other = self.resources.reserve('image', 'image', host_bytes=96)
        other.release()
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(ResourceCancelled):
            adapter.generate([], cancel_event=cancelled)
        self.assertEqual(self.created, 1)
        self.fail_build = True
        with self.assertRaises(ValueError):
            adapter.generate([])
        self.assertFalse(adapter.is_resident)
        self.assertEqual(self.resources.snapshot()['reservations'], {})
        self.fail_build = False
        self.assertEqual(adapter.generate([]), 'answer-fixture-3')
        adapter.close()

    def test_close_failure_preserves_handle_for_retry_and_explicit_close_is_final(self):
        adapter = self.build()
        self.fail_close = True
        with self.assertRaises(RuntimeError):
            self.resources.reserve('image', 'image', host_bytes=96)
        self.assertTrue(adapter.is_resident)
        self.assertIsNotNone(self.references[0]())
        self.fail_close = False
        adapter.close()
        self.assertEqual(self.resources.snapshot()['reservations'], {})
        with self.assertRaises(RuntimeError):
            adapter.generate([])
        self.assertEqual(self.created, 1)
