"""Tiny in-memory Hub fixtures; never fetch real weights."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from api.services.model_catalog import CATALOG, CatalogEntry, CheckpointSource
from api.services.model_downloads import Cancelled, ModelManager, fetch_checkpoint_manifest, source_plan


PRIMARY = 'a' * 40
AUX = 'b' * 40
ENTRY = CatalogEntry('composed-fixture', 'fixture/quant', PRIMARY, 'apache-2.0', 1,
    kind='image', layout='components', source_files=('quant.safetensors',),
    required_files=('model_index.json', 'transformer/config.json', 'vae/config.json'),
    weight_paths=('', 'vae'), auxiliary_sources=(CheckpointSource('fixture/base', AUX,
        files=('model_index.json', 'transformer/config.json'), component_paths=('vae',)),))
PAYLOADS = {
    'fixture/quant': {'quant.safetensors': b'quant', 'other.safetensors': b'excluded'},
    'fixture/base': {'model_index.json': b'{}', 'transformer/config.json': b'{}',
        'transformer/diffusion_pytorch_model.safetensors': b'excluded BF16',
        'vae/config.json': b'{}', 'vae/diffusion_pytorch_model.safetensors': b'vae'},
}


def metadata(repo, revision):
    return {'sha': revision, 'siblings': [dict(rfilename=name, lfs={
        'size': len(value), 'sha256': hashlib.sha256(value).hexdigest()})
        for name, value in PAYLOADS[repo].items()]}


class MultisourceDownloadsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manager = ModelManager(Path(self.temp.name))
        self.catalog = patch.dict(CATALOG, {ENTRY.id: ENTRY})
        self.catalog.start()
        self.addCleanup(self.catalog.stop)
        self.requests = []

    def network(self, url, **kwargs):
        self.requests.append(url)
        if '/api/models/' in url:
            repo = url.split('/api/models/')[1].split('/revision/')[0]
            revision = PRIMARY if repo == ENTRY.repo_id else AUX
            return io.BytesIO(json.dumps(metadata(repo, revision)).encode())
        repo, tail = url.split('huggingface.co/')[1].split('/resolve/')
        revision, name = tail.split('/', 1)
        self.assertEqual(revision, PRIMARY if repo == ENTRY.repo_id else AUX)
        return io.BytesIO(PAYLOADS[repo][name])

    def download(self, opener=None):
        with patch('api.services.model_downloads.urlopen', side_effect=opener or self.network):
            self.manager.start(ENTRY.id)
            self.manager._thread.join(5)
            self.assertFalse(self.manager._thread.is_alive())

    def status(self):
        return next(item for item in self.manager.status()['models'] if item['id'] == ENTRY.id)

    def test_combined_bundle_provenance_exact_bytes_and_exclusion(self):
        self.download()
        self.assertEqual(self.status()['status'], 'complete')
        entry, root = self.manager.get_checkpoint(ENTRY.id)
        marker = json.loads((root / 'complete.json').read_text())
        self.assertEqual(marker['source_plan'], source_plan(entry))
        self.assertEqual(len(marker['files']), 5)
        self.assertEqual(self.status()['total_bytes'], 14)
        self.assertEqual(self.status()['downloaded_bytes'], 14)
        self.assertFalse((root / 'transformer/diffusion_pytorch_model.safetensors').exists())
        self.assertEqual({item['repo_id'] for item in marker['files']}, set(PAYLOADS))
        self.assertEqual(len(self.requests), 7)
        self.assertTrue(all('/api/models/' in url for url in self.requests[:2]))
        restored = ModelManager(Path(self.temp.name))
        self.assertEqual(restored.get_checkpoint(ENTRY.id)[1], root)

    def test_collision_and_missing_selection_rejected_before_weights(self):
        for source in (replace(ENTRY.auxiliary_sources[0], files=('quant.safetensors',)),
                       replace(ENTRY.auxiliary_sources[0], files=('missing.json',))):
            entry = replace(ENTRY, auxiliary_sources=(source,))
            with patch.dict(PAYLOADS['fixture/base'], {'quant.safetensors': b'duplicate'}), patch(
                'api.services.model_downloads.urlopen', side_effect=self.network):
                with self.assertRaises(ValueError): fetch_checkpoint_manifest(entry)
        self.assertTrue(all('/api/models/' in url for url in self.requests))

    def test_auxiliary_revision_mismatch(self):
        def wrong(url, **kwargs):
            if '/api/models/fixture/base/' in url:
                return io.BytesIO(json.dumps(metadata('fixture/base', 'c' * 40)).encode())
            return self.network(url, **kwargs)
        self.download(wrong)
        self.assertEqual(self.status()['status'], 'failed')
        self.assertFalse(self.manager._checkpoint_directory(ENTRY).exists())

    def test_integrity_failure_and_retry(self):
        def corrupt(url, **kwargs):
            if '/resolve/' in url and url.endswith('vae/config.json'):
                return io.BytesIO(b'xx')
            return self.network(url, **kwargs)
        self.download(corrupt)
        self.assertEqual(self.status()['status'], 'failed')
        self.assertFalse(self.manager._checkpoint_directory(ENTRY).exists())
        self.download()
        self.assertEqual(self.status()['status'], 'complete')

    def test_cancel_auxiliary_stream_and_retry_releases_lease(self):
        entered, release = threading.Event(), threading.Event()
        class Blocking(io.BytesIO):
            def read(self, size=-1):
                entered.set()
                release.wait(3)
                return super().read(size)
        def block(url, **kwargs):
            if '/resolve/' in url and url.endswith('vae/config.json'):
                return Blocking(b'{}')
            return self.network(url, **kwargs)
        with patch('api.services.model_downloads.urlopen', side_effect=block):
            self.manager.start(ENTRY.id)
            self.assertTrue(entered.wait(2))
            self.manager.cancel(ENTRY.id)
            release.set()
            self.manager._thread.join(5)
        self.assertEqual(self.status()['status'], 'cancelled')
        self.assertFalse(self.manager._checkpoint_directory(ENTRY).exists())
        self.assertFalse(list(Path(self.temp.name).glob('*.partial')))
        self.download()
        self.assertEqual(self.status()['status'], 'complete')

    def test_cancel_between_metadata_sources(self):
        cancel = threading.Event()
        def cancel_after_first(url, **kwargs):
            result = self.network(url, **kwargs)
            cancel.set()
            return result
        with patch('api.services.model_downloads.urlopen', side_effect=cancel_after_first):
            with self.assertRaises(Cancelled): fetch_checkpoint_manifest(ENTRY, cancel)
        self.assertEqual(len(self.requests), 1)

    def test_marker_source_and_plan_tampering_rejected(self):
        self.download()
        root = self.manager.get_checkpoint(ENTRY.id)[1]
        path = root / 'complete.json'
        baseline = path.read_text()
        for change in ('repo_id', 'revision', 'source_plan'):
            marker = json.loads(baseline)
            if change == 'source_plan': marker[change] = 'invalid'
            else: marker['files'][0][change] = 'unapproved'
            path.write_text(json.dumps(marker))
            with self.assertRaises(ValueError): self.manager.get_checkpoint(ENTRY.id)
        changed = replace(ENTRY, auxiliary_sources=(replace(ENTRY.auxiliary_sources[0], revision='c' * 40),))
        self.assertNotEqual(self.manager._checkpoint_directory(changed), root)

    def test_disk_admission_counts_all_sources(self):
        with patch('api.services.model_downloads.shutil.disk_usage') as disk:
            disk.return_value.free = 1024**3 + 13
            self.download()
        self.assertEqual(self.status()['status'], 'failed')
        self.assertEqual(len(self.requests), 2)

    def test_auth_scoped_to_auxiliary_source(self):
        entry = replace(ENTRY, auxiliary_sources=(replace(ENTRY.auxiliary_sources[0], requires_auth=True),))
        with patch.dict(CATALOG, {ENTRY.id: entry}), patch(
            'api.services.model_downloads.open_gated_checkpoint', side_effect=self.network) as gated:
            self.download()
            self.assertEqual(self.status()['status'], 'complete')
        self.assertEqual(gated.call_count, 5)
        self.assertTrue(all('fixture/base/' in call.args[0] for call in gated.call_args_list))
