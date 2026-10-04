import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from api.services.model_catalog import CATALOG, allowed_asset
from api.services.model_downloads import BusyError, ModelManager, _manifest


def fixture():
    payloads = {
        'config.json': b'{}', 'tokenizer_config.json': b'{}',
        'model.safetensors.index.json': json.dumps({'weight_map': {'weight': 'model.safetensors'}}).encode(),
        'model.safetensors': b'controlled test weights, never inference',
    }
    files = [{'name': name, 'size': len(value), 'algorithm': 'sha256',
              'digest': hashlib.sha256(value).hexdigest()} for name, value in payloads.items()]
    return payloads, files


class ModelDownloadsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manager = ModelManager(Path(self.temp.name))

    def download(self, source=None):
        payloads, files = fixture()
        with patch('api.services.model_downloads._manifest', return_value=files), patch(
            'api.services.model_downloads.urlopen', side_effect=source or (lambda url, **kw: io.BytesIO(payloads[url.rsplit('/', 1)[-1]]))
        ):
            self.manager.start('small')
            self.manager._thread.join(5)
            self.assertFalse(self.manager._thread.is_alive())

    def test_integrity_completion_selection_restart_and_runtime_lease(self):
        with self.assertRaises(ValueError): self.manager.select('small')
        self.download()
        self.assertEqual(self.manager.status()['models'][0]['status'], 'complete')
        self.manager.select('small')
        restored = ModelManager(Path(self.temp.name))
        entry, path = restored.acquire_runtime_model()
        self.assertEqual(entry.id, 'small')
        self.assertTrue(path.is_absolute())
        with self.assertRaises(BusyError): restored.select('small')
        restored.release_runtime_model()
        restored.select('small')
        with self.assertRaises(BusyError): self.manager.start('small')
        (path / 'config.json').write_text('corrupt size')
        with self.assertRaises(ValueError): restored.get_selected()
        self.assertEqual(self.manager.status()['models'][0]['status'], 'failed')

    def test_failure_never_publishes_and_retry_works(self):
        self.download(lambda *args, **kwargs: io.BytesIO(b'corrupt'))
        self.assertEqual(self.manager.status()['models'][0]['status'], 'failed')
        self.assertFalse(self.manager._path(CATALOG['small']).exists())
        self.download()
        self.assertEqual(self.manager.status()['models'][0]['status'], 'complete')

    def test_cancel_and_concurrent_download_conflict(self):
        payloads, files = fixture()
        entered, release = threading.Event(), threading.Event()
        def manifest(entry):
            entered.set()
            release.wait(3)
            return files
        with patch('api.services.model_downloads._manifest', side_effect=manifest):
            self.manager.start('small')
            self.assertTrue(entered.wait(2))
            with self.assertRaises(BusyError): self.manager.start('medium')
            other = ModelManager(Path(self.temp.name))
            with self.assertRaises(BusyError): other.start('medium')
            self.manager.cancel('small')
            release.set()
            self.manager._thread.join(5)
        self.assertEqual(self.manager.status()['models'][0]['status'], 'cancelled')
        self.assertFalse(list(Path(self.temp.name).glob('*.partial')))

    def test_cancel_during_stream_removes_partial_files(self):
        payloads, files = fixture()
        entered, release = threading.Event(), threading.Event()
        class BlockingStream(io.BytesIO):
            def read(self, size=-1):
                entered.set()
                release.wait(3)
                return super().read(size)
        with patch('api.services.model_downloads._manifest', return_value=files), patch(
            'api.services.model_downloads.urlopen',
            side_effect=lambda url, **kwargs: BlockingStream(payloads[url.rsplit('/', 1)[-1]])
        ):
            self.manager.start('small')
            self.assertTrue(entered.wait(2))
            self.manager.cancel('small')
            release.set()
            self.manager._thread.join(5)
        self.assertEqual(self.manager.status()['models'][0]['status'], 'cancelled')
        self.assertFalse(self.manager._path(CATALOG['small']).exists())
        self.assertFalse((Path(self.temp.name) / '.small.partial').exists())

    def test_catalog_allowlist_excludes_duplicate_and_executable_assets(self):
        for name in ('original/model.safetensors', 'metal/model.bin', '../config.json', 'model.py', 'weights.bin'):
            self.assertFalse(allowed_asset(name))
        self.assertTrue(allowed_asset('model_mtp.safetensors'))
        self.assertTrue(allowed_asset('model-00014-of-00014.safetensors'))
        with self.assertRaises(ValueError): self.manager.start('../../outside')

    def test_manifest_rejects_unpinned_revision(self):
        with patch('api.services.model_downloads._json_url', return_value={'sha': 'wrong'}):
            with self.assertRaisesRegex(ValueError, 'revision'): _manifest(CATALOG['small'])

    def test_disk_preflight_stops_before_weight_request(self):
        _, files = fixture()
        with patch('api.services.model_downloads._manifest', return_value=files), patch(
            'api.services.model_downloads.shutil.disk_usage'
        ) as usage, patch('api.services.model_downloads.urlopen') as network:
            usage.return_value.free = 0
            self.manager.start('large')
            self.manager._thread.join(5)
            network.assert_not_called()
        self.assertEqual(self.manager.status()['models'][2]['status'], 'failed')
