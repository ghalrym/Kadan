import base64
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api.server import app
from api.services.video_inputs import VideoInputs


class WanI2VRouteTests(unittest.TestCase):
    def test_uploaded_image_resolves_to_server_path_for_native_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            inputs = VideoInputs(Path(temporary))
            client = TestClient(app)
            with patch('api.routes.v1.videos.inputs.video_inputs', inputs), patch(
                    'api.routes.v1.videos.generations.video_inputs', inputs), patch(
                    'api.routes.v1.videos.generations.video_jobs.submit') as submit:
                submit.return_value = dict(id='fixture', prompt='move', duration='1s', resolution='480p',
                    aspect='wide', fps='16', progress=0, time='now', status='Queued', thumbnail='', progressText='Queued')
                image = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==')
                upload = client.post('/v1/videos/inputs?kind=image', content=image, headers={'Content-Type': 'image/png'})
                self.assertEqual(upload.status_code, 200)
                identifier = upload.json()['id']
                response = client.post('/v1/videos/generations', json=dict(model='wan22-i2v-a14b', prompt='move',
                    image_id=identifier, duration=1, fps=16, resolution='480p'))
                self.assertEqual(response.status_code, 202, response.text)
                model, spec = submit.call_args.args
                self.assertEqual(model, 'wan22-i2v-a14b')
                self.assertEqual(Path(spec.image_path).read_bytes(), image)
                response = client.post('/v1/videos/generations', json=dict(model=model, prompt='move', image_id='/etc/passwd'))
                self.assertEqual(response.status_code, 422)
                self.assertEqual(submit.call_count, 1)
