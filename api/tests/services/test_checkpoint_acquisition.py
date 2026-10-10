"""Mocked download ownership only: no network, model loading or inference."""
import asyncio
import fcntl
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from api.inference.cancellation import run_cancellable_thread
from api.inference.resources import ResourceCancelled
from api.services.model_downloads import ModelManager
from api.tests.services.test_model_downloads import fixture


class CheckpointAcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.folder = self.enterContext(tempfile.TemporaryDirectory())
        self.store = ModelManager(Path(self.folder))
        self.payloads, self.files = fixture()
        self.enterContext(patch('api.services.model_downloads.fetch_checkpoint_manifest', return_value=self.files))
        self.enterContext(patch('api.services.model_downloads.urlopen', side_effect=
            lambda url, **kw: io.BytesIO(self.payloads[url.rsplit('/', 1)[-1]])))

    def test_verifies_publishes_and_reuses_shared_directory(self):
        entry, path = self.store.ensure_checkpoint('small', threading.Event())
        self.assertEqual((entry, path), self.store.get_checkpoint('small'))
        with patch('api.services.model_downloads.urlopen', side_effect=AssertionError('unexpected download')):
            self.assertEqual(path, self.store.ensure_checkpoint('small', threading.Event())[1])
        self.assertFalse((self.store.root / '.small.partial').exists())

    def test_cancelled_request_joins_its_writer_and_releases_lock(self):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        def manifest(entry):
            entered.set()
            if not release.wait(3): raise AssertionError('test synchronization timeout')
            return self.files
        async def run():
            task = asyncio.create_task(run_cancellable_thread(self.store.ensure_checkpoint, 'small'))
            self.assertTrue(await asyncio.to_thread(entered.wait, 3))
            task.cancel()
            await asyncio.sleep(.01)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError): await task
            finished.set()
        with patch('api.services.model_downloads.fetch_checkpoint_manifest', side_effect=manifest):
            asyncio.run(run())
        self.assertTrue(finished.is_set())
        self.assertFalse(self.store._thread.is_alive())
        self.assertFalse((self.store.root / '.small.partial').exists())
        self.assertEqual(self.store._jobs['small'].status, 'cancelled')
        with (self.store.root / '.download.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_waiter_cannot_cancel_independently_owned_download(self):
        entered, release, cancel = threading.Event(), threading.Event(), threading.Event()
        def manifest(entry):
            entered.set()
            if not release.wait(3): raise AssertionError('test synchronization timeout')
            return self.files
        with patch('api.services.model_downloads.fetch_checkpoint_manifest', side_effect=manifest):
            self.store.start('small')
            self.assertTrue(entered.wait(3))
            original_join = self.store._thread.join
            def join(timeout=None):
                cancel.set()
                original_join(0)
            with patch.object(self.store._thread, 'join', side_effect=join):
                with self.assertRaises(ResourceCancelled): self.store.ensure_checkpoint('small', cancel)
            self.assertFalse(self.store._cancel.is_set())
            release.set()
            self.store._thread.join(3)
        self.assertEqual(self.store._jobs['small'].status, 'complete')

    def test_cross_process_writer_wait_is_cancellable(self):
        cancel = threading.Event()
        with (self.store.root / '.download.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(cancel, 'wait', side_effect=lambda _: cancel.set()):
                with self.assertRaises(ResourceCancelled): self.store.ensure_checkpoint('small', cancel)
        self.assertIsNone(self.store._thread)

    def test_integrity_failure_is_not_retried_or_published(self):
        with patch('api.services.model_downloads.urlopen', return_value=io.BytesIO(b'bad')):
            with self.assertRaisesRegex(ValueError, 'Integrity verification|Unexpected file size'):
                self.store.ensure_checkpoint('small', threading.Event())
        self.assertFalse(self.store._checkpoint_directory(self.store._catalog_entry('small')).exists())

    def test_publication_between_initial_check_and_writer_lock(self):
        with patch.object(self.store, '_checkpoint_complete', side_effect=[False, True]), patch.object(self.store, '_download') as download:
            from api.services.model_downloads import BusyError
            with self.assertRaises(BusyError): self.store.start('small')
            download.assert_not_called()
        with (self.store.root / '.download.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
