"""Exercise actual HTTP conditioning upload and server-side ID resolution."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.server import app
from api.services.video_inputs import VideoInputs
from api.services.video_jobs import VideoJobs


class VideoInputRoutesTests(unittest.TestCase):
    def test_uploaded_image_resolves_without_exposing_local_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = VideoInputs(Path(directory) / 'inputs')
            seen = []
            class Provider:
                def validate(self, spec):
                    seen.append(spec)
                def generate(self, spec, path, cancellation):
                    path.write_bytes(b'video fixture')
            jobs = VideoJobs(Path(directory) / 'output', lambda _: Provider())
            with patch('api.routes.v1.videos.inputs.video_inputs', inputs), patch('api.routes.v1.videos.generations.video_inputs', inputs), patch('api.routes.v1.videos.generations.video_jobs', jobs):
                client = TestClient(app)
                uploaded = client.post('/v1/videos/inputs?kind=image', content=b'image fixture', headers={'content-type': 'image/png'})
                self.assertEqual(uploaded.status_code, 200)
                self.assertEqual(set(uploaded.json()), {'id'})
                response = client.post('/v1/videos/generations', json={'model': 'wan22-ti2v-5b', 'prompt': 'forest', 'image_id': uploaded.json()['id']})
                self.assertEqual(response.status_code, 202)
                self.assertEqual(Path(seen[0].image_path).read_bytes(), b'image fixture')
                bad = client.post('/v1/videos/generations', json={'model': 'wan22-ti2v-5b', 'prompt': 'forest', 'image_id': '/etc/passwd'})
                self.assertEqual(bad.status_code, 422)
            jobs.close()
