import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from api.inference.qwen_speech import QwenSpeechProvider, QwenSpeechSession
from api.inference.resources import ResourceCancelled, ResourceManager
from api.inference.speech import SpeechInput, SpeechPlan, SpeechRegistry, SpeechRuntime, SpeechUnavailable
from api.inference.speech_process import OwnedSpeechProcess
from api.services.qwen_tts_catalog import SPEECH_MODELS


QWEN_FIXTURE = '''
from pathlib import Path
class Qwen3TTSModel:
    @classmethod
    def from_pretrained(cls, checkpoint, **options):
        assert options['local_files_only'] is True
        root = Path(checkpoint)
        count = root / 'loads'
        count.write_text(str(int(count.read_text()) + 1) if count.exists() else '1')
        return cls()
    def generate_voice_design(self, **options):
        return [[0] * 240], 24000
    def generate_custom_voice(self, **options):
        return [[0] * 240], 24000
    def generate_voice_clone(self, **options):
        return [[0] * 240], 24000
'''
SOUNDFILE_FIXTURE = '''
import wave
def write(path, samples, rate, **kwargs):
    with wave.open(path, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b'\\0\\0' * len(samples))
'''


@unittest.skipUnless(os.name == 'posix', 'Owned worker groups require POSIX')
class ResidentWorkerTests(unittest.TestCase):
    def test_real_worker_protocol_loads_once_for_two_requests_and_reaps_on_unload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, content in {'qwen_tts.py': QWEN_FIXTURE, 'soundfile.py': SOUNDFILE_FIXTURE,
                                  'torch.py': "float32 = 'float32'\nbfloat16 = 'bfloat16'\n",
                                  'numpy.py': ''}.items():
                (root / name).write_text(content)
            resources = ResourceManager(1000, {})
            registry = SpeechRegistry()
            provider = QwenSpeechProvider()
            session = QwenSpeechSession(sys.executable, root, 'cpu', 'fixture')
            registry.register(provider)
            runtime = SpeechRuntime(registry, lambda: resources)
            request = SpeechInput('Hello', {'mode': 'describe', 'description': 'warm'}, model_id='qwen-tts-1.7b-design')
            with patch.dict(os.environ, PYTHONPATH=str(root)), \
                 patch.object(provider, 'enabled', return_value=True), \
                 patch.object(provider, 'prepare', return_value=SpeechPlan(('fixture',), 400, lambda: session)):
                try:
                    for _ in range(2):
                        result = runtime.generate(request, threading.Event())
                        self.assertTrue(result.wav.startswith(b'RIFF'))
                        state = next(iter(resources.snapshot()['reservations'].values()))
                        self.assertEqual(state['active_leases'], 0)
                    process = session.worker.process
                    resident_directory = Path(session.directory.name)
                    self.assertIsNone(process.poll())
                    self.assertEqual((root / 'loads').read_text(), '1')
                    self.assertEqual(list(resident_directory.glob('request-*')), [])
                finally:
                    runtime.unload()
                self.assertIsNotNone(process.poll())
                self.assertFalse(resident_directory.exists())
                self.assertEqual(resources.snapshot()['reservations'], {})

    def test_cancellation_reaps_actual_process_before_releasing_accounting(self):
        resources = ResourceManager(1000, {})
        registry = SpeechRegistry()
        provider = QwenSpeechProvider()
        session = QwenSpeechSession(sys.executable, Path('/unused'), 'cpu', 'fixture')
        registry.register(provider)
        runtime = SpeechRuntime(registry, lambda: resources)
        event = threading.Event()
        launched = threading.Event()
        process = OwnedSpeechProcess()
        errors = []
        with tempfile.TemporaryDirectory() as directory, open(os.devnull, 'wb') as log:
            def load(cancel):
                process.start([sys.executable, '-c', 'import time; time.sleep(60)'], env=os.environ.copy(), log=log)
                session.worker = process
                launched.set()
                session._wait(Path(directory) / 'never-ready', cancel)
            original_stop = process.stop
            def stop():
                self.assertTrue(resources.snapshot()['reservations'])
                original_stop()
                self.assertTrue(resources.snapshot()['reservations'])
            process.stop = stop
            request = SpeechInput('Hello', {'mode': 'describe', 'description': 'warm'}, model_id='qwen-tts-1.7b-design')
            with patch.object(provider, 'enabled', return_value=True), \
                 patch.object(provider, 'prepare', return_value=SpeechPlan(('fixture',), 400, lambda: session)), \
                 patch.object(session, 'load', side_effect=load):
                def generate():
                    try:
                        runtime.generate(request, event)
                    except BaseException as exc:
                        errors.append(exc)
                thread = threading.Thread(target=generate)
                thread.start()
                try:
                    self.assertTrue(launched.wait(3))
                    event.set()
                    thread.join(8)
                    self.assertFalse(thread.is_alive())
                    self.assertIsInstance(errors[0], ResourceCancelled)
                    self.assertIsNone(process.process)
                    self.assertEqual(resources.snapshot()['reservations'], {})
                finally:
                    event.set()
                    runtime.unload()

    @unittest.skipUnless(Path('/proc/self/stat').exists(), 'Linux descendant state verification')
    def test_owned_process_reaps_descendants_even_after_parent_exits(self):
        with tempfile.TemporaryDirectory() as directory, open(os.devnull, 'wb') as log:
            pid_path = Path(directory) / 'child'
            program = "import subprocess,sys,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
            process = OwnedSpeechProcess()
            process.start([sys.executable, '-c', program, str(pid_path)], env=os.environ.copy(), log=log)
            try:
                process.process.wait(timeout=5)
                self.assertTrue(process._alive())
                process.stop()
                self.assertIsNone(process.process)
                child_stat = Path('/proc') / pid_path.read_text() / 'stat'
                if child_stat.exists():
                    self.assertEqual(child_stat.read_text().rsplit(')', 1)[1].split()[0], 'Z')
            finally:
                process.stop()


class QwenAdapterTests(unittest.TestCase):
    def test_validation_and_capabilities_stay_in_adapter(self):
        provider = QwenSpeechProvider()
        models = {item.id: item for item in provider.models()}
        self.assertTrue(models['qwen-tts-1.7b-custom'].supports_instruction)
        self.assertFalse(models['qwen-tts-0.6b-custom'].supports_instruction)
        self.assertIn('Ryan', models['qwen-tts-1.7b-custom'].speakers)
        self.assertEqual(models['qwen-tts-1.7b-custom'].default_speaker, 'Ryan')
        for request in (
            SpeechInput('Hi', {'mode': 'custom', 'speaker': 'Ryan', 'instruction': 'warm'}, model_id='qwen-tts-0.6b-custom'),
            SpeechInput('Hi', {'mode': 'custom', 'speaker': 'Ada'}, model_id='qwen-tts-1.7b-custom'),
            SpeechInput('Hi', {'mode': 'clone', 'sample': 'UklGRg=='}, model_id='qwen-tts-1.7b-base'),
            SpeechInput('Hi', {'mode': 'clone', 'sample': 'https://example.org', 'speaker_only': True}, model_id='qwen-tts-1.7b-base'),
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                provider.validate(request)

    def test_prepare_checks_local_revision_and_estimates_host_and_device(self):
        model = SPEECH_MODELS['qwen-tts-1.7b-design']
        provider = QwenSpeechProvider()
        request = SpeechInput('Hi', {'mode': 'describe', 'description': 'warm'}, model_id=model.id)
        with patch.dict(os.environ, KADAN_QWEN_TTS_PYTHON=sys.executable, KADAN_QWEN_TTS_DEVICE='cuda:1'), \
             patch('api.inference.qwen_speech.model_manager.get_checkpoint', return_value=(model, Path('/local')), create=True):
            plan = provider.prepare(request)
            self.assertEqual(plan.host_bytes, model.estimated_bytes * 3 + 2 * 1024**3)
            self.assertEqual(plan.device_bytes, {1: plan.host_bytes})
            self.assertIn(model.revision, plan.identity)
            self.assertIsNone(plan.create().worker.process)
