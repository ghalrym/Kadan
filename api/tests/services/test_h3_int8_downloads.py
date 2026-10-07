"""Composite downloads retain integrity and do not accept the old BF16 marker."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api.services.model_catalog import CATALOG, allowed_asset
from api.services.model_downloads import ModelManager, fetch_checkpoint_manifest


class H3Int8Downloads(unittest.TestCase):
    def test_manifest_is_pinned_and_excludes_duplicate_weights_and_code(self):
        entry = CATALOG['h3-fl2va-int8-turbo']
        files = fetch_checkpoint_manifest(entry)
        self.assertEqual(sum(item['name'].endswith('.safetensors') for item in files), 5)
        self.assertFalse(CATALOG['h3-fl2va'].inference_available)
        self.assertLess(entry.estimated_bytes, 70_000_000_000)
        for item in files:
            self.assertRegex(item['url'], r'^https://huggingface.co/[^/]+/[^/]+/resolve/[0-9a-f]{40}/')
            self.assertFalse(item['name'].endswith(('.py', '.bin')))
        for name in ('../outside', '/weights', 'FL2VA/transformer/model-00001-of-00013.safetensors'):
            self.assertFalse(allowed_asset(name, entry))

    def test_cross_repository_publication_and_tampered_marker_rejected(self):
        payloads = {'config.json': b'{}', 'weights/model.safetensors': b'quantized'}
        files = tuple(dict(name=name, size=len(data), digest=hashlib.sha256(data).hexdigest(),
                           algorithm='sha256', url=f'https://example.test/{index}/{name}')
                      for index, (name, data) in enumerate(payloads.items()))
        entry = replace(CATALOG['h3-fl2va-int8-turbo'], manifest=files,
                        required_files=tuple(payloads), estimated_bytes=11)
        with tempfile.TemporaryDirectory() as directory, patch.dict(CATALOG, {entry.id: entry}):
            manager = ModelManager(Path(directory))
            by_url = {item['url']: payloads[item['name']] for item in files}
            with patch('api.services.model_downloads.urlopen', side_effect=lambda url, **kw: io.BytesIO(by_url[url])) as read:
                manager.start(entry.id, license_acknowledged=True)
                manager._thread.join(5)
                self.assertFalse(manager._thread.is_alive())
                self.assertEqual(read.call_count, 2)
            _, checkpoint = manager.get_checkpoint(entry.id)
            marker = json.loads((checkpoint / 'complete.json').read_text())
            marker['files'][0]['digest'] = '0' * 64
            (checkpoint / 'complete.json').write_text(json.dumps(marker))
            with self.assertRaisesRegex(ValueError, 'completely'):
                manager.get_checkpoint(entry.id)
