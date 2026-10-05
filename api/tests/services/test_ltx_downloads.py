"""The LTX bundle contains exactly the five official distilled BF16 components."""
from io import BytesIO
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch

from api.inference.ltx import ASSETS
from api.services.model_catalog import CATALOG, allowed_asset, validate_assets
from api.services.model_downloads import ModelManager


class LTXDownloadTests(unittest.TestCase):
    def test_exact_bundle_and_completed_resolver(self):
        entry = CATALOG['ltx-2.5-distilled']
        names = set(ASSETS.values())
        self.assertEqual(set(entry.required_files), names)
        for name in names:
            self.assertTrue(allowed_asset(name, entry))
        self.assertFalse(allowed_asset('diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors', entry))
        with self.assertRaises(ValueError):
            validate_assets(entry, names - {ASSETS['audio_vae_path']})
        payload = b'tiny fixture, not a model'
        manifest = [dict(name=name, size=len(payload), digest=hashlib.sha256(payload).hexdigest(), algorithm='sha256') for name in names]
        with tempfile.TemporaryDirectory() as directory:
            manager = ModelManager(Path(directory))
            with patch('api.services.model_downloads.fetch_checkpoint_manifest', return_value=manifest), patch('api.services.model_downloads.urlopen', side_effect=lambda *_args, **_kwargs: BytesIO(payload)):
                manager.start(entry.id, license_acknowledged=True)
                manager._thread.join(2)
            resolved, path = manager.get_checkpoint(entry.id)
            self.assertEqual(resolved, entry)
            for name in names:
                self.assertEqual((path / name).read_bytes(), payload)
            manager.close()
