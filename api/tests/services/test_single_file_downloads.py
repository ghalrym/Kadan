import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api.services.model_catalog import CATALOG, CatalogEntry, allowed_asset
from api.services.model_downloads import ModelManager, fetch_checkpoint_manifest


class SingleFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manager = ModelManager(Path(self.temp.name))
        self.payload = b'controlled checkpoint fixture'
        self.entry = CatalogEntry('whisper-fixture', 'openai/whisper', hashlib.sha256(self.payload).hexdigest(),
            'mit', 0, kind='transcription', layout='single_file', asset_name='fixture.pt',
            asset_url='https://example.com/pinned/fixture.pt', required_files=('fixture.pt',))
        registry = patch.dict(CATALOG, {self.entry.id: self.entry})
        registry.start()
        self.addCleanup(registry.stop)
        self.requests = []

    def source(self, request, **kwargs):
        self.requests.append(request)
        response = io.BytesIO(self.payload)
        response.headers = {'Content-Length': str(len(self.payload))}
        return response

    def download(self):
        with patch('api.services.model_downloads.urlopen', side_effect=self.source):
            self.manager.start(self.entry.id)
            self.manager._thread.join(5)
        self.assertFalse(self.manager._thread.is_alive())

    def test_head_then_verified_atomic_publication(self):
        self.download()
        self.assertEqual(self.requests[0].get_method(), 'HEAD')
        self.assertEqual(self.requests[1], self.entry.asset_url)
        entry, directory = self.manager.get_checkpoint(self.entry.id)
        self.assertEqual(entry, self.entry)
        self.assertEqual((directory / 'fixture.pt').read_bytes(), self.payload)
        self.assertFalse((self.manager.root / '.whisper-fixture.partial').exists())
        self.assertIsNone(next(model for model in self.manager.status()['models'] if model['id'] == self.entry.id)['context_limit'])
        self.assertFalse(allowed_asset('../fixture.pt', entry))
        self.assertFalse(allowed_asset('other.pt', entry))

    def test_missing_head_size_fails_before_body_request(self):
        response = io.BytesIO()
        response.headers = {}
        with patch('api.services.model_downloads.urlopen', return_value=response) as network:
            with self.assertRaisesRegex(ValueError, 'size metadata'):
                fetch_checkpoint_manifest(self.entry)
        self.assertEqual(network.call_count, 1)

    def test_hash_mismatch_never_publishes_and_retry_succeeds(self):
        original = self.payload
        self.payload = b'corrupted'
        self.download()
        with self.assertRaises(ValueError): self.manager.get_checkpoint(self.entry.id)
        self.payload = original
        self.download()
        self.manager.get_checkpoint(self.entry.id)
