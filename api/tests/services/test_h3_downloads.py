"""Synthetic component bundles exercise publication without downloading weights."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api.services.model_catalog import CATALOG, allowed_asset, validate_assets
from api.services.model_downloads import ModelManager, fetch_checkpoint_manifest


class H3DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manager = ModelManager(Path(self.temp.name))
        self.entry = CATALOG['h3-fl2va']
        self.payloads = {name: b'{}' for name in self.entry.required_files}
        for directory in self.entry.weight_paths:
            self.payloads[f'{directory}/model.safetensors'] = b'fixture'
        self.payloads['FL2VA/transformer/model.safetensors.index.json'] = json.dumps(
            {'weight_map': {'weight': 'model.safetensors'}}).encode()

    def download(self):
        files = [{'name': name, 'size': len(value), 'algorithm': 'sha256',
                  'digest': hashlib.sha256(value).hexdigest()} for name, value in self.payloads.items()]
        with patch('api.services.model_downloads.fetch_checkpoint_manifest', return_value=files), patch(
            'api.services.model_downloads.urlopen', side_effect=lambda url, **kwargs:
                io.BytesIO(self.payloads[url.split(self.entry.revision + '/')[1]])):
            self.manager.start(self.entry.id, license_acknowledged=True)
            self.manager._thread.join(5)
            self.assertFalse(self.manager._thread.is_alive())

    def test_explicit_acknowledgement_before_any_io(self):
        with patch('api.services.model_downloads.urlopen') as network:
            with self.assertRaisesRegex(ValueError, 'explicitly continue'):
                self.manager.start(self.entry.id)
            network.assert_not_called()
            self.assertFalse((self.manager.root / '.download.lock').exists())

    def test_nested_integrity_publication_and_no_llm_selection(self):
        self.download()
        entry, path = self.manager.get_checkpoint(self.entry.id)
        self.assertEqual(entry, self.entry)
        self.assertTrue((path / 'FL2VA/video_vae/source/model.safetensors').is_file())
        self.assertIsNone(self.manager.status()['selected_model_id'])
        with self.assertRaisesRegex(ValueError, 'not a language model'):
            self.manager.select(entry.id)
        with self.assertRaises(ValueError): self.manager.set_context(entry.id, 8192)
        with self.assertRaises(ValueError): self.manager.configured_context(entry.id)
        self.assertIsNone(self.manager.status()['models'][-1]['context_limit'])

    def test_nested_index_missing_shards_fail_before_publish(self):
        self.payloads['FL2VA/transformer/model.safetensors.index.json'] = b'{"weight_map":{"w":"../../outside.safetensors"}}'
        self.download()
        self.assertEqual(self.manager.status()['models'][-1]['status'], 'failed')
        self.assertFalse(self.manager._checkpoint_directory(self.entry).exists())

    def test_nested_directory_symlink_invalidates_completed_checkpoint(self):
        self.download()
        _, path = self.manager.get_checkpoint(self.entry.id)
        directory = path / 'FL2VA/audio_vae'
        directory.rename(path / 'moved')
        directory.symlink_to(path / 'moved', target_is_directory=True)
        with self.assertRaises(ValueError): self.manager.get_checkpoint(self.entry.id)

    def test_allowlist_excludes_other_family_duplicates_code_and_traversal(self):
        for name in ('Ref2VA/transformer/model.safetensors', 'transformer/model.safetensors',
                     'FL2VA/transformer/model.py', 'FL2VA/../config.json', '/config.json',
                     'FL2VA//transformer/model.safetensors'):
            self.assertFalse(allowed_asset(name, self.entry), name)
        self.assertTrue(allowed_asset('FL2VA/video_vae/source/model.safetensors', self.entry))
        with self.assertRaises(ValueError): validate_assets(self.entry, set(self.payloads) - {'FL2VA/audio_vae/model.safetensors'})

    def test_upstream_manifest_selects_only_complete_requested_bundle(self):
        siblings = [{'rfilename': name, 'lfs': {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}}
                    for name, data in self.payloads.items()]
        siblings.append({'rfilename': 'Ref2VA/transformer/model.safetensors'})
        with patch('api.services.model_downloads.urlopen', return_value=io.BytesIO(json.dumps(
                {'sha': self.entry.revision, 'siblings': siblings}).encode())):
            files = fetch_checkpoint_manifest(self.entry)
        self.assertEqual({item['name'] for item in files}, set(self.payloads))
