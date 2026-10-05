"""Opaque conditioning upload IDs cannot select arbitrary local files."""
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from api.services.video_inputs import VideoInputs


class VideoInputTests(unittest.IsolatedAsyncioTestCase):
    async def test_upload_publish_and_resolve(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = VideoInputs(Path(directory))
            async def chunks():
                yield b'tiny image fixture'
            identifier = await inputs.save('image', chunks(), 'image/png')
            self.assertEqual(Path(inputs.resolve(identifier, 'image')).read_bytes(), b'tiny image fixture')
            with self.assertRaises(ValueError):
                inputs.resolve(identifier, 'audio')
            with self.assertRaises(ValueError):
                inputs.resolve('../secret', 'image')

    async def test_empty_upload_not_published(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs = VideoInputs(Path(directory))
            async def chunks():
                yield b''
            with self.assertRaises(ValueError):
                await inputs.save('image', chunks(), 'image/png')
            self.assertEqual(list((Path(directory) / 'image').iterdir()), [])
