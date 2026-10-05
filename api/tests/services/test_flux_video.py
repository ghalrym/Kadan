from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from api.services.flux_video import FluxVideoProvider


def spec(**changes):
    return SimpleNamespace(**(dict(prompt='a fox', aspect='16:9', resolution='720p', duration=8,
                                  fps=24, seed=None, negative_prompt='') | changes))


class FluxVideoTests(unittest.TestCase):
    def test_documented_payload_and_completed_mp4(self):
        client = Mock()
        client.generate.return_value = 'https://media.example.test/video'
        client.download.side_effect = lambda url, path, cancel, limit: path.write_bytes(b'\x00\x00\x00\x18ftypmp42')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'video.mp4'
            FluxVideoProvider(client).generate(spec(), path, threading.Event())
            self.assertTrue(path.exists())
        endpoint, payload, cancel = client.generate.call_args.args
        self.assertEqual(endpoint, 'flux-3-video')
        self.assertEqual(payload, dict(prompt='a fox', aspect_ratio='16:9', resolution='hd', duration=8,
                                       generate_audio=True, mode='t2v'))

    def test_unsupported_controls_do_not_submit(self):
        client = Mock()
        for changes in (dict(seed=42), dict(negative_prompt='bad'), dict(fps=30), dict(duration=4), dict(duration=8.5)):
            with self.assertRaises(ValueError):
                FluxVideoProvider(client).validate(spec(**changes))
        client.generate.assert_not_called()

    def test_invalid_media_is_removed(self):
        client = Mock()
        client.download.side_effect = lambda url, path, cancel, limit: path.write_bytes(b'not video')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'video.mp4'
            with self.assertRaisesRegex(RuntimeError, 'invalid MP4'):
                FluxVideoProvider(client).generate(spec(), path, threading.Event())
            self.assertFalse(path.exists())
