"""Download only synthetic bytes for the exact TI2V component bundle."""
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api.services.model_catalog import CATALOG, allowed_asset
from api.services.model_downloads import ModelManager


class TI2VDownloadTests(unittest.TestCase):
    def test_exact_bundle_atomic_completion(self):
        entry = CATALOG['wan22-ti2v-5b']
        self.assertFalse(allowed_asset('assets/example.mp4', entry))
        files = {name: b'fixture' for name in entry.required_files}
        files['diffusion_pytorch_model.safetensors.index.json'] = json.dumps({'weight_map': {
            f'weight{i}': f'diffusion_pytorch_model-0000{i}-of-00003.safetensors' for i in (1, 2, 3)}}).encode()
        manifest = [dict(name=name, size=len(payload), digest=hashlib.sha256(payload).hexdigest(), algorithm='sha256') for name, payload in files.items()]
        def open_file(url, **kwargs):
            return BytesIO(files[str(url).split('/resolve/' + entry.revision + '/')[1]])
        with tempfile.TemporaryDirectory() as directory:
            manager = ModelManager(Path(directory))
            with patch('api.services.model_downloads.fetch_checkpoint_manifest', return_value=manifest), patch('api.services.model_downloads.urlopen', side_effect=open_file):
                manager.start(entry.id)
                manager._thread.join(2)
            resolved, path = manager.get_checkpoint(entry.id)
            self.assertEqual(resolved, entry)
            self.assertEqual((path / 'Wan2.2_VAE.pth').read_bytes(), b'fixture')
            manager.close()
