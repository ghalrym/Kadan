import base64
import io
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave

from api.inference.resources import ResourceManager, ResourceCancelled
from api.services import speech
from api.services.qwen_tts_catalog import SPEECH_MODELS


class SpeechWorkerTests(unittest.TestCase):
    def run_worker(self, cancel=False, failure=False):
        model = SPEECH_MODELS['qwen-tts-1.7b-design']
        resources = ResourceManager(100 * 1024**3, {})
        event = threading.Event()
        process = SimpleNamespace(returncode=1 if failure else 0)
        if cancel:
            event.set()
        def start(args, **kwargs):
            self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
            self.assertTrue(resources.snapshot()['reservations'])
            with wave.open(args[-1], 'wb') as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(24000)
                out.writeframes(b'\0\0' * 240)
            process.poll = lambda: process.returncode
            return process
        with tempfile.NamedTemporaryFile() as python, \
             patch.dict('os.environ', KADAN_QWEN_TTS_PYTHON=python.name, KADAN_QWEN_TTS_DEVICE='cpu'), \
             patch.object(speech, 'ENABLED_SPEECH_MODELS', {model.id}), \
             patch.object(speech.manager, 'get_checkpoint', return_value=(model, Path('/verified')), create=True), \
             patch.object(speech.runtime_manager, 'ensure_resources', return_value=resources), \
             patch.object(speech.subprocess, 'Popen', side_effect=start):
            try:
                return speech.generate_speech(dict(model_id=model.id, script='hello', language='Auto', voice={'mode': 'describe', 'description': 'warm'}), event)
            finally:
                self.assertEqual(resources.snapshot()['reservations'], {})

    def test_complete_waveform_and_release(self):
        result = self.run_worker()
        self.assertTrue(base64.b64decode(result['audio_base64']).startswith(b'RIFF'))

    def test_failure_releases(self):
        with self.assertRaises(speech.SpeechUnavailable):
            self.run_worker(failure=True)

    def test_cancel_before_allocation(self):
        with self.assertRaises(ResourceCancelled):
            self.run_worker(cancel=True)

    def test_disconnect_reaps_child_before_releasing_lease(self):
        model = SPEECH_MODELS['qwen-tts-1.7b-design']
        resources = ResourceManager(100 * 1024**3, {})
        event = threading.Event()
        class Process:
            alive = True
            def poll(self):
                return None if self.alive else -15
            def terminate(self):
                self.assert_owned()
            def assert_owned(self):
                assert resources.snapshot()['reservations']
            def wait(self, timeout=None):
                self.assert_owned()
                self.alive = False
        process = Process()
        def start(*args, **kwargs):
            event.set()
            return process
        with tempfile.NamedTemporaryFile() as python, \
             patch.dict('os.environ', KADAN_QWEN_TTS_PYTHON=python.name, KADAN_QWEN_TTS_DEVICE='cpu'), \
             patch.object(speech, 'ENABLED_SPEECH_MODELS', {model.id}), \
             patch.object(speech.manager, 'get_checkpoint', return_value=(model, Path('/verified')), create=True), \
             patch.object(speech.runtime_manager, 'ensure_resources', return_value=resources), \
             patch.object(speech.subprocess, 'Popen', side_effect=start):
            with self.assertRaises(ResourceCancelled):
                speech.generate_speech({'model_id': model.id}, event)
        self.assertFalse(process.alive)
        self.assertEqual(resources.snapshot()['reservations'], {})
