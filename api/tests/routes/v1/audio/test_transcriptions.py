import unittest
from fastapi.testclient import TestClient
from api.server import app


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_valid_requests_truthfully_report_unavailable(self):
        for formatting in (True, False):
            response = self.client.post('/v1/audio/transcriptions', json={'audio': 'example-audio-reference', 'formatting': formatting})
            self.assertEqual(response.status_code, 503)
            self.assertIn('no speech-to-text provider', response.json()['detail'])
            self.assertNotIn('text', response.json())

    def test_invalid_references_and_extra_fields_are_rejected(self):
        for body in ({}, {'audio': ''}, {'audio': '   '}, {'audio': 'x' * 2049}, {'audio': 'ref', 'file': 'fake-upload'}):
            self.assertEqual(self.client.post('/v1/audio/transcriptions', json=body).status_code, 422)
