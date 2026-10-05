"""Only synthetic component bytes are transferred in download tests."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api.services.model_catalog import CATALOG, allowed_asset, validate_assets
from api.services.model_downloads import ModelManager


class QwenImageDownloadTests(unittest.TestCase):
    def test_bundle_acknowledgement_integrity_and_component_indexes(self):
        entry = CATALOG['qwen-image-2.1']
        payloads = {name: b'{}' for name in entry.required_files}
        for component in entry.weight_paths:
            payloads[f'{component}/model.safetensors'] = b'fixture'
        for name in payloads:
            if name.endswith('.index.json'):
                payloads[name] = json.dumps({'weight_map': {'fixture': 'model.safetensors'}}).encode()
        validate_assets(entry, set(payloads))
        self.assertFalse(allowed_asset('assets/example.png', entry))
        self.assertFalse(allowed_asset('transformer/custom.py', entry))
        with tempfile.TemporaryDirectory() as folder:
            manager = ModelManager(Path(folder))
            with patch('api.services.model_downloads.urlopen') as network:
                with self.assertRaisesRegex(ValueError, 'explicitly continue'):
                    manager.start(entry.id)
                network.assert_not_called()
            files = [{'name': name, 'size': len(data), 'algorithm': 'sha256', 'digest': hashlib.sha256(data).hexdigest()}
                     for name, data in payloads.items()]
            with patch('api.services.model_downloads.fetch_checkpoint_manifest', return_value=files), patch(
                'api.services.model_downloads.urlopen', side_effect=lambda url, **kwargs: BytesIO(payloads[url.split(entry.revision + '/')[1]])):
                manager.start(entry.id, license_acknowledged=True)
                manager._thread.join(5)
            resolved, path = manager.get_checkpoint(entry.id)
            self.assertEqual(resolved.revision, 'd26bb61231c349cf6b7896fa83353113880e1ba3')
            self.assertTrue((path / 'vae/model.safetensors').is_file())
            self.assertIsNone(manager.status()['selected_model_id'])
            with self.assertRaises(ValueError):
                manager.select(entry.id)
