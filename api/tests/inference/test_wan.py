"""Shared native Wan process lifecycle without model weights."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from api.inference.wan import WanProvider
from api.inference.video import VideoSpec
from api.inference.resources import ResourceCancelled, ResourceManager


class WanTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / 'fixture').write_bytes(b'fixture')
        self.manager = Mock()
        self.manager.get_checkpoint.return_value = (SimpleNamespace(kind='video'), self.root)
        self.resources = ResourceManager(10000, {0: 10000})
        self.provider = WanProvider('test', 't2v-A14B', self.manager, self.resources, '/usr/bin/python3')

    def tearDown(self):
        self.directory.cleanup()

    def test_frame_rate_checked_before_launch(self):
        with self.assertRaisesRegex(ValueError, '16 fps'):
            self.provider.validate(VideoSpec('hello', fps=24))
        self.provider.validate(VideoSpec('hello', negative_prompt='blur', fps=16))

    def test_cancel_reaps_process_group_before_releasing_memory(self):
        event = threading.Event()
        process = Mock(pid=123)
        process.poll.return_value = None
        def start(*args, **kwargs):
            self.assertTrue(kwargs['start_new_session'])
            event.set()
            return process
        with patch('api.inference.wan.subprocess.Popen', side_effect=start), patch('api.inference.wan.os.killpg') as kill:
            with self.assertRaises(ResourceCancelled):
                self.provider.generate(VideoSpec('hello', fps=16), self.root / 'out.mp4', event)
        kill.assert_called_once()
        process.wait.assert_called_once()
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    def test_text_checkpoint_rejects_image_conditioning(self):
        with self.assertRaisesRegex(ValueError, 'does not accept an image'):
            self.provider.validate(VideoSpec('hello', fps=16, image_path='source.png'))
