from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.server import app
from api.services.model_downloads import BusyError, ModelManager


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
        self.assertEqual([item['id'] for item in response.json()['models']], ['small', 'medium', 'large', 'h3-fl2va', 'qwen-image-2.1'])
        self.assertEqual(self.client.put('/v1/models/selection', json={'model_id': 'small'}).status_code, 400)
        self.assertEqual(self.client.put('/v1/models/selection', json={'model_id': 'arbitrary'}).status_code, 422)
        self.assertEqual(self.client.post('/v1/models/arbitrary/download').status_code, 400)
        self.assertEqual(self.client.delete('/v1/models/small/download').status_code, 409)

    def test_mutation_routes_preserve_error_responses(self):
        routes = [
            ('set_context', 'PUT', '/v1/models/small/context', {'context_limit': 4096}),
            ('select', 'PUT', '/v1/models/selection', {'model_id': 'small'}),
            ('start', 'POST', '/v1/models/small/download', None),
            ('cancel', 'DELETE', '/v1/models/small/download', None),
        ]
        errors = [(BusyError('model is busy'), 409), (ValueError('invalid model state'), 400),
                  (OSError('secret local path'), 503)]
        for operation, method, path, body in routes:
            # Status reads stay inside the same error boundary as the operation.
            for failing_method in (operation, 'status'):
                for error, expected in errors:
                    with self.subTest(operation=operation, failing_method=failing_method, expected=expected):
                        with patch.object(self.manager, operation), patch.object(
                            self.manager, failing_method, side_effect=error
                        ):
                            response = self.client.request(method, path, json=body)
                        self.assertEqual(response.status_code, expected)
                        if expected == 503:
                            self.assertIn('permissions', response.json()['detail'])
                            self.assertNotIn('secret', response.text)
                        else:
                            self.assertEqual(response.json()['detail'], str(error))
