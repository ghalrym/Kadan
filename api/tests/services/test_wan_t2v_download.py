"""Wan original checkpoint integrity with synthetic bytes only."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api.services.model_catalog import CATALOG, allowed_asset, validate_assets
from api.services.model_downloads import ModelManager


class WanT2VDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manager = ModelManager(Path(self.temp.name))
        self.entry = CATALOG['wan22-t2v-a14b']
        self.payloads = {name: b'{}' for name in self.entry.required_files}
        for expert in self.entry.weight_paths:
            self.payloads[f'{expert}/diffusion_pytorch_model.safetensors'] = b'fixture'
            self.payloads[f'{expert}/diffusion_pytorch_model.safetensors.index.json'] = json.dumps(
                {'weight_map': {'weight': 'diffusion_pytorch_model.safetensors'}}).encode()

    def test_original_bundle_requires_both_experts_encoder_vae_and_tokenizer(self):
        validate_assets(self.entry, set(self.payloads))
        for missing in ('Wan2.1_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth',
                        'google/umt5-xxl/spiece.model',
                        'high_noise_model/diffusion_pytorch_model.safetensors',
                        'low_noise_model/diffusion_pytorch_model.safetensors'):
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                validate_assets(self.entry, set(self.payloads) - {missing})

    def test_only_reviewed_pickle_files_are_allowed(self):
        self.assertTrue(allowed_asset('Wan2.1_VAE.pth', self.entry))
        self.assertTrue(allowed_asset('models_t5_umt5-xxl-enc-bf16.pth', self.entry))
        for filename in ('unknown.pth', 'model.py', 'assets/example.mp4',
                         'high_noise_model/unknown.pth', '../Wan2.1_VAE.pth'):
            self.assertFalse(allowed_asset(filename, self.entry))

    def test_atomic_download_resolves_for_native_provider_without_llm_selection(self):
        manifest = [{'name': name, 'size': len(value), 'algorithm': 'sha256',
                     'digest': hashlib.sha256(value).hexdigest()} for name, value in self.payloads.items()]
        with patch('api.services.model_downloads.fetch_checkpoint_manifest', return_value=manifest), \
             patch('api.services.model_downloads.urlopen', side_effect=lambda url, **kwargs:
                   io.BytesIO(self.payloads[url.split(self.entry.revision + '/')[1]])):
            self.manager.start(self.entry.id)
            self.manager._thread.join(5)
            self.assertFalse(self.manager._thread.is_alive())
        entry, checkpoint = self.manager.get_checkpoint(self.entry.id)
        self.assertEqual(entry.revision, 'c8c270b13ee05bfa474194ac9fb07a5868a97cea')
        self.assertTrue((checkpoint / 'Wan2.1_VAE.pth').is_file())
        self.assertIsNone(self.manager.status()['selected_model_id'])
