"""Complete Animate model plus pose/mask assets, using synthetic bytes only."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api.services.model_catalog import CATALOG, allowed_asset, validate_assets
from api.services.model_downloads import ModelManager


class AnimateDownloadTests(unittest.TestCase):
    def test_auxiliary_assets_required_and_duplicate_formats_excluded(self):
        entry = CATALOG['wan22-animate-14b']
        files = set(entry.required_files) | {'diffusion_pytorch_model.safetensors'}
        validate_assets(entry, files)
        for path in ('process_checkpoint/det/yolov10m.onnx', 'process_checkpoint/sam2/sam2_hiera_large.pt',
                     'process_checkpoint/pose2d/vitpose_h_wholebody.onnx', 'relighting_lora.ckpt'):
            self.assertTrue(allowed_asset(path, entry))
            with self.assertRaises(ValueError):
                validate_assets(entry, files - {path})
        for path in ('xlm-roberta-large/model.safetensors', 'xlm-roberta-large/pytorch_model.bin',
                     'relighting_lora/adapter_model.safetensors', 'process_checkpoint/sam2/sam2_hiera_small.pt',
                     'process_checkpoint/sam2/download_ckpts.sh', 'process_checkpoint/FLUX.1-Kontext-dev/model.safetensors'):
            self.assertFalse(allowed_asset(path, entry), path)

    def test_original_bundle_download_is_atomic_and_native_resolvable(self):
        entry = CATALOG['wan22-animate-14b']
        values = {name: b'{}' for name in entry.required_files}
        values['diffusion_pytorch_model.safetensors'] = b'fixture'
        values['diffusion_pytorch_model.safetensors.index.json'] = json.dumps(
            {'weight_map': {'weight': 'diffusion_pytorch_model.safetensors'}}).encode()
        files = [{'name': name, 'size': len(value), 'algorithm': 'sha256',
                  'digest': hashlib.sha256(value).hexdigest()} for name, value in values.items()]
        with tempfile.TemporaryDirectory() as directory:
            manager = ModelManager(Path(directory))
            with patch('api.services.model_downloads.fetch_checkpoint_manifest', return_value=files), \
                 patch('api.services.model_downloads.urlopen', side_effect=lambda url, **kwargs:
                       io.BytesIO(values[url.split(entry.revision + '/')[1]])):
                manager.start(entry.id)
                manager._thread.join(5)
                self.assertFalse(manager._thread.is_alive())
            actual, checkpoint = manager.get_checkpoint(entry.id)
            self.assertEqual(actual.revision, 'cb93a225fbaf1ca100f54e79da8f994995b689b3')
            self.assertTrue((checkpoint / 'process_checkpoint/det/yolov10m.onnx').is_file())
            self.assertIsNone(manager.status()['selected_model_id'])
