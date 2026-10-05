import unittest
from fastapi.testclient import TestClient
from api.server import app


class VideoRoutesTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_no_fabricated_history(self):
        self.assertEqual(self.client.get('/v1/videos').json(), {'jobs': []})
        for video_id in ('vid_3f9a21', 'vid_77c0e4', 'missing'):
            self.assertEqual(self.client.get('/v1/videos/' + video_id).status_code, 404)

    def test_submit_fails_without_provider_and_does_not_queue(self):
        for _ in range(2):
            result = self.client.post('/v1/videos/generations', json={'prompt': 'Clouds over mountains'})
            self.assertEqual(result.status_code, 503)
            self.assertIn('No job was queued', result.json()['detail'])
            self.assertEqual(self.client.get('/v1/videos').json(), {'jobs': []})

    def test_invalid_requests_rejected_before_provider(self):
        for body in ({'prompt': ' '}, {'prompt': 'x', 'duration': 0},
                     {'prompt': 'x', 'fps': 121}, {'prompt': 'x', 'aspect': 'wide'},
                     {'prompt': 'x', 'unexpected': True}):
            self.assertEqual(self.client.post('/v1/videos/generations', json=body).status_code, 422)
