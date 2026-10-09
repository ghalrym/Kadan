from unittest.mock import patch
from api.memory_manager import memory_manager
from api.tests.memory_manager.helpers import direct_feature
import unittest
from api.services.runtime import RuntimeFailure
from fastapi.testclient import TestClient
from api.server import app


class ImageEndpointsTest(unittest.TestCase):
    def setUpFeature(self):
        self.enterContext(patch.object(memory_manager, 'submit', direct_feature(memory_manager, 'image')))

    def setUp(self):
        self.setUpFeature()
        self.client = TestClient(app)

    def test_no_fixture_history(self):
        response = self.client.get('/v1/images')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'images': []})

    def test_missing_checkpoint_fails_truthfully(self):
        with patch('api.services.image_jobs.image_jobs.generate', side_effect=RuntimeFailure('Download Qwen-Image-2.1', 409)):
            for path, body in [('generations', {'prompt': 'A tree', 'count': 1}),
                               ('edits', {'prompt': 'A tree', 'image': 'inline-source'})]:
                response = self.client.post('/v1/images/' + path, json=body)
                self.assertEqual(response.status_code, 409)
                self.assertIn('Download Qwen', response.json()['detail'])
                self.assertNotIn('image', response.json())

    def test_reject_invalid_inputs_before_provider_dispatch(self):
        for body in [{'prompt': '   '}, {'prompt': 'x', 'count': 3}, {'prompt': 'x', 'seed': -1}]:
            self.assertEqual(self.client.post('/v1/images/generations', json=body).status_code, 422)
