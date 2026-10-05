"""LTX subprocess/lease tests without optional dependencies or checkpoints."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from api.inference.ltx import ASSETS, LTXProvider, MODEL_REVISION
from api.inference.resources import ResourceCancelled, ResourceManager
from api.inference.video import VideoSpec


class LTXTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ASSETS.values():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixture')
        manager = Mock()
        manager.get_checkpoint.return_value = (SimpleNamespace(revision=MODEL_REVISION), self.root)
        self.resources = ResourceManager(10000, {0: 10000})
        self.provider = LTXProvider(manager, self.resources, '/usr/bin/python3')
        self.output = self.root / 'output.mp4'

    def tearDown(self):
        self.temp.cleanup()

    def test_rejects_unsupported_negative_prompt_before_launch(self):
        with self.assertRaisesRegex(ValueError, 'negative prompt'):
            self.provider.validate(VideoSpec('hello', negative_prompt='blur'))

    def test_process_failure_releases_reservation(self):
        process = Mock()
        process.poll.return_value = 1
        process.returncode = 1
        with patch('api.inference.ltx.subprocess.Popen', return_value=process):
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                self.provider.generate(VideoSpec('hello'), self.output, threading.Event())
        self.assertEqual(self.resources.snapshot()['reservations'], {})
        self.assertIsNone(self.resources.snapshot()['exclusive_owner'])

    def test_cancel_terminates_and_reaps_before_release(self):
        process = Mock()
        process.poll.return_value = None
        event = threading.Event()
        def start(*args, **kwargs):
            event.set()
            self.assertEqual(len(self.resources.snapshot()['reservations']), 1)
            return process
        with patch('api.inference.ltx.subprocess.Popen', side_effect=start):
            with self.assertRaises(ResourceCancelled):
                self.provider.generate(VideoSpec('hello'), self.output, event)
        process.terminate.assert_called_once()
        process.wait.assert_called_once()
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    def test_missing_assets_fail_before_launch(self):
        (self.root / ASSETS['audio_vae_path']).unlink()
        with self.assertRaisesRegex(RuntimeError, 'components'):
            self.provider.validate(VideoSpec('hello'))
