"""Lazy service initialization and process-wide admission ownership."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from api.inference.stt import model as transcription, catalog as whisper_catalog


class LazyManagerTests(unittest.TestCase):
    def setUp(self):
        transcription._cached_transcription_manager.cache_clear()
        self.addCleanup(transcription._cached_transcription_manager.cache_clear)

    def test_import_does_not_construct_manager(self):
        subprocess.run([sys.executable, '-c',
            'from api.inference.stt.model import _cached_transcription_manager; '
            'assert _cached_transcription_manager.cache_info().currsize == 0'], check=True)

    def test_concurrent_first_requests_share_one_manager_and_admission_lock(self):
        barrier = threading.Barrier(8)
        real_manager = transcription.TranscriptionManager
        def construct_slowly():
            # Release the GIL while first construction is in flight, exposing
            # duplicate construction if the lock moves inside lru_cache.
            time.sleep(0.05)
            return real_manager()
        def first_request():
            barrier.wait(timeout=5)
            return transcription.get_transcription_manager()
        with patch.object(transcription, 'TranscriptionManager', side_effect=construct_slowly) as construct:
            construct.assert_not_called()
            with ThreadPoolExecutor(max_workers=8) as pool:
                managers = list(pool.map(lambda _: first_request(), range(8)))
            construct.assert_called_once_with()
        self.assertTrue(all(manager is managers[0] for manager in managers))
        self.assertIsNone(managers[0].resources)
        self.assertIsNone(managers[0].factory)
        managers[0].lock.acquire()
        try:
            self.assertFalse(transcription.get_transcription_manager().lock.acquire(blocking=False))
        finally:
            managers[0].lock.release()
        self.assertTrue(transcription.get_transcription_manager().lock.acquire(blocking=False))
        managers[0].lock.release()

    def test_failed_construction_can_retry(self):
        manager = object()
        with patch.object(transcription, 'TranscriptionManager', side_effect=[RuntimeError('setup failed'), manager]):
            with self.assertRaisesRegex(RuntimeError, 'setup failed'):
                transcription.get_transcription_manager()
            self.assertIs(transcription.get_transcription_manager(), manager)
            self.assertIs(transcription.get_transcription_manager(), manager)


class LazyCatalogTests(unittest.TestCase):
    def setUp(self):
        whisper_catalog.get_whisper_checkpoints.cache_clear()
        self.addCleanup(whisper_catalog.get_whisper_checkpoints.cache_clear)

    def test_import_does_not_read_registrations(self):
        subprocess.run([sys.executable, '-c',
            'from unittest.mock import patch; '
            'guard = patch("pathlib.Path.glob", side_effect=AssertionError("eager scan")); '
            'guard.start(); '
            'from api.inference.stt.catalog import get_whisper_checkpoints; '
            'assert get_whisper_checkpoints.cache_info().currsize == 0'], check=True)

    def test_existing_registration_path_is_cached_until_explicit_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            services = Path(directory)
            registrations = services / 'checkpoints'
            registrations.mkdir()
            (registrations / 'tiny.json').write_text('{"name":"tiny"}')
            with patch.object(whisper_catalog, '__file__', str(services / 'catalog.py')):
                first = whisper_catalog.get_whisper_checkpoints()
                self.assertEqual(list(first), ['tiny'])
                (registrations / 'base.json').write_text('{"name":"base"}')
                self.assertIs(whisper_catalog.get_whisper_checkpoints(), first)
                self.assertEqual(list(first), ['tiny'])
                whisper_catalog.get_whisper_checkpoints.cache_clear()
                self.assertEqual(list(whisper_catalog.get_whisper_checkpoints()), ['base', 'tiny'])

    def test_invalid_registration_failure_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            registration = Path(directory) / 'model.json'
            registration.write_text('{"name":"turbo"}')
            with patch.object(Path, 'glob', return_value=[registration]):
                with self.assertRaisesRegex(ValueError, 'canonical'):
                    whisper_catalog.get_whisper_checkpoints()
                registration.write_text('{"name":"large-v3-turbo"}')
                self.assertEqual(list(whisper_catalog.get_whisper_checkpoints()), ['large-v3-turbo'])
