import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.server import app
from api.services.model_downloads import ModelManager, BusyError
from api.services.model_catalog import CATALOG


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manager = ModelManager(Path(self.directory.name))

    def test_defaults_and_persistence_across_restart(self):
        self.assertEqual(self.manager.configured_context('small'), 65536)
        self.manager.set_context('small', 131072)
        restarted = ModelManager(self.manager.root)
        self.assertEqual(restarted.configured_context('small'), 131072)
        self.assertEqual(restarted.configured_context('medium'), 65536)
        restarted.set_context('small', None)
        self.assertIsNone(ModelManager(self.manager.root).configured_context('small'))

    def test_legacy_custom_and_null_settings_are_preserved_without_rewriting(self):
        path = self.manager.root / 'context.json'
        original = '{"small":4096,"medium":null}'
        path.write_text(original)
        restarted = ModelManager(self.manager.root)
        self.assertEqual(restarted.configured_context('small'), 4096)
        self.assertIsNone(restarted.configured_context('medium'))
        self.assertEqual(restarted.configured_context('large'), 65536)
        self.assertEqual([model['context_limit'] for model in restarted.status()['models'] if model['kind'] == 'llm'], [4096, None, 65536])
        self.assertEqual(path.read_text(), original)

    def test_default_is_never_silently_clamped_to_a_smaller_verified_limit(self):
        with patch.object(self.manager, 'architecture_context', return_value=32768):
            self.assertEqual(self.manager.configured_context('small'), 65536)
            self.assertEqual(self.manager.status()['models'][0]['context_limit'], 65536)
            with self.assertRaisesRegex(ValueError, 'architecture maximum'):
                self.manager.set_context('small', 65536)
            self.assertFalse((self.manager.root / 'context.json').exists())
            self.manager.set_context('small', 32768)
            self.assertEqual(self.manager.configured_context('small'), 32768)

    def test_api_rejects_oversized_presets_and_keeps_saved_custom_value(self):
        self.manager.set_context('small', 4096)
        with patch('api.routes.v1.models.model_manager', self.manager), \
                patch.object(self.manager, 'architecture_context', return_value=131072):
            client = TestClient(app)
            for value in (500000, 1000000):
                response = client.put('/v1/models/small/context', json={'context_limit': value})
                self.assertEqual(response.status_code, 400)
                self.assertEqual(self.manager.configured_context('small'), 4096)
            for value in (8192, 16384, 32768, 65536, 131072):
                response = client.put('/v1/models/small/context', json={'context_limit': value})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(ModelManager(self.manager.root).configured_context('small'), value)

    def test_positive_strict_integer_and_busy(self):
        for value in [0, -1, True, 1.5, '4096', 2**31]:
            with self.assertRaises(ValueError):
                self.manager.set_context('small', value)
        self.manager._in_use = True
        with self.assertRaises(BusyError):
            self.manager.set_context('small', 4096)
        self.assertFalse((self.manager.root / 'context.json').exists())

    def test_completed_checkpoint_architecture_maximum(self):
        path = self.manager._checkpoint_directory(CATALOG['small'])
        path.mkdir()
        (path / 'config.json').write_text(json.dumps({'text_config': {'max_position_embeddings': 262144}}))
        with patch.object(self.manager, '_checkpoint_complete', return_value=True):
            self.assertEqual(self.manager.architecture_context('small'), 262144)
            self.manager.set_context('small', 262144)
            with self.assertRaisesRegex(ValueError, 'architecture maximum'):
                self.manager.set_context('small', 262145)
        self.assertEqual(self.manager.configured_context('small'), 262144)

    def test_api_rejects_invalid_and_busy_and_exposes_saved_value(self):
        with patch('api.routes.v1.models.model_manager', self.manager):
            client = TestClient(app)
            for value in [True, 0, 3.5, '32']:
                self.assertEqual(client.put('/v1/models/small/context', json={'context_limit': value}).status_code, 422)
            response = client.put('/v1/models/small/context', json={'context_limit': 131072})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['models'][0]['context_limit'], 131072)
            self.assertIsNone(response.json()['models'][0]['architecture_context_limit'])
            self.manager._in_use = True
            self.assertEqual(client.put('/v1/models/small/context', json={'context_limit': None}).status_code, 409)
