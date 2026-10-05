import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api.services.model_catalog import CATALOG, allowed_asset, validate_assets
from api.services.model_downloads import ModelManager


class WanI2VCatalogTests(unittest.TestCase):
    def test_pinned_components_and_atomic_fixture_download(self):
        entry = CATALOG['wan22-i2v-a14b']
        self.assertEqual(entry.revision, '206a9ee1b7bfaaf8f7e4d81335650533490646a3')
        self.assertEqual(entry.repo_id, 'Wan-AI/Wan2.2-I2V-A14B')
        payloads = {name: b'{}' for name in entry.required_files}
        for directory in entry.weight_paths:
            payloads[f'{directory}/diffusion_pytorch_model.safetensors'] = b'fixture weights'
            payloads[f'{directory}/diffusion_pytorch_model.safetensors.index.json'] = json.dumps(
                {'weight_map': {'fixture': 'diffusion_pytorch_model.safetensors'}}).encode()
        validate_assets(entry, set(payloads))
        for name in payloads:
            self.assertTrue(allowed_asset(name, entry), name)
        for name in ('examples/input.jpg', 'nohup.out', 'README.py', '../Wan2.1_VAE.pth'):
            self.assertFalse(allowed_asset(name, entry), name)
        manifest = [{'name': name, 'size': len(data), 'digest': hashlib.sha256(data).hexdigest(),
                     'algorithm': 'sha256'} for name, data in payloads.items()]
        with tempfile.TemporaryDirectory() as temporary:
            manager = ModelManager(Path(temporary))
            with patch('api.services.model_downloads.fetch_checkpoint_manifest', return_value=manifest), patch(
                    'api.services.model_downloads.urlopen', side_effect=lambda url, **kwargs:
                    io.BytesIO(payloads[url.split(entry.revision + '/')[1]])):
                manager.start(entry.id)
                manager._thread.join(5)
            self.assertFalse(manager._thread.is_alive())
            _, path = manager.get_checkpoint(entry.id)
            self.assertTrue((path / 'google/umt5-xxl/spiece.model').is_file())
            self.assertTrue((path / 'Wan2.1_VAE.pth').is_file())

    def test_both_experts_are_required(self):
        entry = CATALOG['wan22-i2v-a14b']
        with self.assertRaises(ValueError):
            validate_assets(entry, set(entry.required_files) | {'high_noise_model/model.safetensors'})
