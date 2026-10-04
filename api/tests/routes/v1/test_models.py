from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.server import app
from api.services.model_downloads import ModelManager


class ModelsRouteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manager = ModelManager(Path(self.temp.name))
        patched = patch('api.routes.v1.models.model_manager', self.manager)
        patched.start()
        self.addCleanup(patched.stop)
        self.client = TestClient(app)

    def test_catalog_and_validation(self):
        response = self.client.get('/v1/models')
        self.assertEqual(response.status_code, 200)
        self.assertEqual([item['id'] for item in response.json()['models']], ['small', 'medium', 'large'])
        self.assertEqual(self.client.put('/v1/models/selection', json={'model_id': 'small'}).status_code, 400)
        self.assertEqual(self.client.put('/v1/models/selection', json={'model_id': 'arbitrary'}).status_code, 422)
        self.assertEqual(self.client.post('/v1/models/arbitrary/download').status_code, 400)
        self.assertEqual(self.client.delete('/v1/models/small/download').status_code, 409)

    def test_storage_errors_are_actionable(self):
        with patch.object(self.manager, 'start', side_effect=OSError('secret local path')):
            response = self.client.post('/v1/models/small/download')
        self.assertEqual(response.status_code, 503)
        self.assertIn('permissions', response.json()['detail'])
        self.assertNotIn('secret', response.text)
