from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from api.services.model_catalog import CATALOG, allowed_asset
from api.services.model_downloads import BusyError, ModelManager, fetch_checkpoint_manifest


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
        with patch('api.services.model_downloads.fetch_checkpoint_manifest', return_value=files), patch(
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
        self.assertFalse(self.manager._checkpoint_directory(CATALOG['small']).exists())
        self.download()
        self.assertEqual(self.manager.status()['models'][0]['status'], 'complete')

    def test_cancel_and_concurrent_download_conflict(self):
        payloads, files = fixture()
        entered, release = threading.Event(), threading.Event()
        def manifest(entry):
            entered.set()
            release.wait(3)
            return files
        with patch('api.services.model_downloads.fetch_checkpoint_manifest', side_effect=manifest):
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
        with patch('api.services.model_downloads.fetch_checkpoint_manifest', return_value=files), patch(
            'api.services.model_downloads.urlopen',
            side_effect=lambda url, **kwargs: BlockingStream(payloads[url.rsplit('/', 1)[-1]])
        ):
            self.manager.start('small')
            self.assertTrue(entered.wait(2))
            self.manager.cancel('small')
            release.set()
            self.manager._thread.join(5)
        self.assertEqual(self.manager.status()['models'][0]['status'], 'cancelled')
        self.assertFalse(self.manager._checkpoint_directory(CATALOG['small']).exists())
        self.assertFalse((Path(self.temp.name) / '.small.partial').exists())

    def test_catalog_allowlist_excludes_duplicate_and_executable_assets(self):
        for name in ('original/model.safetensors', 'metal/model.bin', '../config.json', 'model.py', 'weights.bin'):
            self.assertFalse(allowed_asset(name))
        self.assertTrue(allowed_asset('model_mtp.safetensors'))
        self.assertTrue(allowed_asset('model-00014-of-00014.safetensors'))
        with self.assertRaises(ValueError): self.manager.start('../../outside')

    def test_manifest_rejects_unpinned_revision(self):
        with patch('api.services.model_downloads.urlopen', return_value=io.BytesIO(json.dumps({'sha': 'wrong', 'siblings': []}).encode())):
            with self.assertRaisesRegex(ValueError, 'revision'): fetch_checkpoint_manifest(CATALOG['small'])

    def test_disk_preflight_stops_before_weight_request(self):
        _, files = fixture()
        with patch('api.services.model_downloads.fetch_checkpoint_manifest', return_value=files), patch(
            'api.services.model_downloads.shutil.disk_usage'
        ) as usage, patch('api.services.model_downloads.urlopen') as network:
            usage.return_value.free = 0
            self.manager.start('large')
            self.manager._thread.join(5)
            network.assert_not_called()
        self.assertEqual(self.manager.status()['models'][2]['status'], 'failed')

    def test_same_length_corruption_fails_digest_verification(self):
        payloads, _ = fixture()
        def corrupt_same_length(url, **kwargs):
            original = payloads[url.rsplit('/', 1)[-1]]
            return io.BytesIO(b'x' * len(original))
        self.download(corrupt_same_length)
        status = self.manager.status()['models'][0]
        self.assertEqual(status['status'], 'failed')
        self.assertIn('Integrity verification failed', status['error'])
        self.assertFalse(self.manager._checkpoint_directory(CATALOG['small']).exists())

    def test_download_worker_start_failure_releases_filesystem_lock(self):
        with patch('api.services.model_downloads.threading.Thread.start', side_effect=RuntimeError('cannot start')):
            with self.assertRaises(RuntimeError):
                self.manager.start('small')
        self.assertEqual(self.manager.status()['models'][0]['status'], 'failed')
        self.manager = ModelManager(Path(self.temp.name))
        self.download()
        self.assertEqual(self.manager.status()['models'][0]['status'], 'complete')

    def test_malformed_upstream_json_is_failed_without_echoing_input(self):
        malformed = {'sha': CATALOG['small'].revision, 'siblings': 'private-token-not-a-list'}
        with patch('api.services.model_downloads.urlopen', return_value=io.BytesIO(json.dumps(malformed).encode())):
            self.manager.start('small')
            self.manager._thread.join(5)
        status = self.manager.status()['models'][0]
        self.assertEqual(status['status'], 'failed')
        self.assertEqual(status['error'], 'Checkpoint metadata is malformed')
        self.assertNotIn('private-token', status['error'])
        self.assertFalse(self.manager._checkpoint_directory(CATALOG['small']).exists())

    def test_corrupt_completion_marker_is_not_selectable(self):
        self.download()
        checkpoint = self.manager._checkpoint_directory(CATALOG['small'])
        for invalid in ({'revision': CATALOG['small'].revision, 'files': [None]},
                        {'revision': CATALOG['small'].revision, 'files': [{'name': [], 'size': 1}]}):
            (checkpoint / 'complete.json').write_text(json.dumps(invalid))
            self.assertEqual(self.manager.status()['models'][0]['status'], 'failed')
            with self.assertRaises(ValueError):
                self.manager.select('small')


class CheckpointManifestTests(unittest.TestCase):
    def upstream_metadata(self):
        payloads, _ = fixture()
        siblings = []
        for filename, payload in payloads.items():
            if filename.endswith('.safetensors'):
                siblings.append({'rfilename': filename, 'lfs': {
                    'size': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}})
            else:
                digest = hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()
                siblings.append({'rfilename': filename, 'size': len(payload), 'blobId': digest})
        return {'sha': CATALOG['small'].revision, 'siblings': siblings}

    def test_typed_upstream_manifest_and_duplicate_formats(self):
        metadata = self.upstream_metadata()
        metadata['siblings'].append({'rfilename': 'original/model.safetensors'})
        with patch('api.services.model_downloads.urlopen', return_value=io.BytesIO(json.dumps(metadata).encode())) as fetch:
            manifest = fetch_checkpoint_manifest(CATALOG['small'])
        self.assertEqual(len(manifest), 4)
        self.assertEqual(manifest[0]['algorithm'], 'git-sha1')
        self.assertEqual(manifest[-1]['algorithm'], 'sha256')
        self.assertIn(CATALOG['small'].revision, fetch.call_args.args[0])

    def test_malformed_metadata_sizes_digests_and_duplicates_rejected(self):
        baseline = self.upstream_metadata()
        invalid_documents = [None, [], {'sha': baseline['sha'], 'siblings': [None]}]
        for key, value in [('size', True), ('size', -1), ('blobId', 'not-a-digest')]:
            invalid = deepcopy(baseline)
            invalid['siblings'][0][key] = value
            invalid_documents.append(invalid)
        duplicate = deepcopy(baseline)
        duplicate['siblings'].append(duplicate['siblings'][0])
        invalid_documents.append(duplicate)
        for metadata in invalid_documents:
            with self.subTest(metadata=metadata), patch(
                'api.services.model_downloads.urlopen', return_value=io.BytesIO(json.dumps(metadata).encode())
            ):
                with self.assertRaises(ValueError):
                    fetch_checkpoint_manifest(CATALOG['small'])

    def test_git_blob_and_lfs_integrity_both_checked_in_download(self):
        payloads, _ = fixture()
        metadata = self.upstream_metadata()
        def source(url, **kwargs):
            if '?blobs=true' in url:
                return io.BytesIO(json.dumps(metadata).encode())
            return io.BytesIO(payloads[url.rsplit('/', 1)[-1]])
        with tempfile.TemporaryDirectory() as directory, patch('api.services.model_downloads.urlopen', side_effect=source):
            manager = ModelManager(Path(directory))
            manager.start('small')
            manager._thread.join(5)
            self.assertEqual(manager.status()['models'][0]['status'], 'complete')
