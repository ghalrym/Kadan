from unittest.mock import patch
from api.memory_manager import memory_manager
from api.tests.memory_manager.helpers import direct_feature
import unittest
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

    def test_generation_and_edit_fail_truthfully(self):
        for path, body in [('generations', {'prompt': 'A tree', 'count': 1}),
                           ('edits', {'prompt': 'A tree', 'image': 'local-reference', 'strength': .5})]:
            response = self.client.post('/v1/images/' + path, json=body)
            self.assertEqual(response.status_code, 503)
            self.assertIn('native image worker' if path == 'generations' else 'No image provider', response.json()['detail'])
            self.assertNotIn('image', response.json())

    def test_reject_invalid_inputs_before_provider_dispatch(self):
        for body in [{'prompt': '   '}, {'prompt': 'x', 'count': 3}, {'prompt': 'x', 'seed': -1}]:
            self.assertEqual(self.client.post('/v1/images/generations', json=body).status_code, 422)
