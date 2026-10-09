import base64
import io
import threading
import unittest
from unittest.mock import patch
import wave

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.inference.resources import ResourceManager, ResourceBusy, ResourceCancelled, ResourceExhausted
from api.inference.tts.speech_runtime import SpeechInput, SpeechModel, SpeechPlan, SpeechRegistry, SpeechResult, SpeechRuntime, SpeechUnavailable
from api.routes.v1.audio.speech import router
from api.services import speech
from api.memory_manager import MemoryManager, memory_manager
from api.inference.tts.speech_requests import SpeechRequests
from api.tests.memory_manager.helpers import direct_feature


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b'\0\0' * 240)
    return buffer.getvalue()


class AlternateProvider:
    """A different architecture with no Qwen catalog, environment or subprocess."""
    def __init__(self, resources, name='alternate', failure=None):
        self.resources, self.name, self.failure = resources, name, failure
        self.loads = self.calls = self.unloads = 0
        self.started = threading.Event()
        self.finish = threading.Event()
        self.block = False
        self.parked = False

    def models(self):
        return (SpeechModel(self.name, 'Alternative voice', 'custom', ('Ada',), True),)

    def enabled(self, model_id):
        return True

    def validate(self, request):
        if request.voice.get('speaker') != 'Ada':
            raise ValueError('Choose Ada')

    def prepare(self, request):
        return SpeechPlan(('revision-1',), 400, lambda: self, {0: 400})

    def load(self, cancel):
        self.parked = False
        self.assert_owned(active=True)
        self.loads += 1
        if self.failure == 'load':
            raise RuntimeError('load failed after allocation')

    def generate(self, request, cancel):
        self.assert_owned(active=True)
        self.calls += 1
        self.started.set()
        if self.block:
            while not self.finish.wait(.01):
                if cancel.is_set():
                    raise ResourceCancelled('cancelled')
        if self.failure == 'generate':
            raise RuntimeError('generation failed')
        return SpeechResult(b'broken' if self.failure == 'audio' else wav_bytes(), self.name)

    def unload(self):
        self.assert_owned(active=False)
        self.unloads += 1
        if self.failure == 'unload':
            raise RuntimeError('cleanup incomplete')

    def offload_to_ram(self):
        self.assert_owned(active=False)
        self.parked = True

    def restore(self, cancel):
        self.parked = False
        self.assert_owned(active=True)

    def assert_owned(self, active):
        states = list(self.resources.snapshot()['reservations'].values())
        assert sum(s['host_bytes'] for s in states) == 400
        assert sum(s['device_bytes'].get(0, 0) for s in states) == (0 if self.parked else 400)
        assert all(s['active_leases'] == int(active) for s in states)


class SpeechLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.resources = ResourceManager(1000, {0: 1000})
        self.provider = AlternateProvider(self.resources)
        self.registry = SpeechRegistry()
        self.registry.register(self.provider)
        self.runtime = SpeechRuntime(self.registry, lambda: self.resources)
        self.request = SpeechInput('Hello', {'mode': 'custom', 'speaker': 'Ada'}, 'Martian', 'alternate')
        self.cancel = threading.Event()

    def tearDown(self):
        self.provider.failure = None
        self.runtime.unload()
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    def test_explicit_load_and_repeated_generation_keep_one_idle_resident(self):
        self.runtime.load(self.request, self.cancel)
        self.provider.assert_owned(active=False)
        for _ in range(2):
            self.assertTrue(self.runtime.generate(self.request, self.cancel).wav.startswith(b'RIFF'))
            self.provider.assert_owned(active=False)
        self.assertEqual((self.provider.loads, self.provider.calls, self.provider.unloads), (1, 2, 0))
        self.runtime.unload()
        self.assertEqual(self.provider.unloads, 1)
        self.runtime.generate(self.request, self.cancel)
        self.assertEqual(self.provider.loads, 2)

    def test_idle_resident_is_evicted_by_other_workload_and_reloaded(self):
        self.runtime.generate(self.request, self.cancel)
        other = self.resources.reserve('image', 'image', 800, {0: 800})
        self.assertEqual(self.provider.unloads, 1)
        self.assertEqual(set(self.resources.snapshot()['reservations']), {'image'})
        other.release()
        self.runtime.generate(self.request, self.cancel)
        self.assertEqual(self.provider.loads, 2)

    def test_load_generation_and_invalid_audio_failures_clean_up(self):
        for phase in ('load', 'generate', 'audio'):
            self.provider.failure = phase
            with self.subTest(phase=phase), self.assertRaises((RuntimeError, SpeechUnavailable)):
                self.runtime.generate(self.request, self.cancel)
            self.assertEqual(self.resources.snapshot()['reservations'], {})

    def test_failed_unload_retains_accounting_until_retry_succeeds(self):
        self.runtime.load(self.request, self.cancel)
        self.provider.failure = 'unload'
        with self.assertRaisesRegex(RuntimeError, 'cleanup incomplete'):
            self.runtime.unload()
        self.provider.assert_owned(active=False)
        with self.assertRaisesRegex(RuntimeError, 'cleanup incomplete'):
            self.resources.reserve('image', 'image', 800, {0: 800})
        self.provider.assert_owned(active=False)
        self.provider.failure = None
        self.runtime.unload()
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    def test_active_work_cannot_be_evicted_and_unload_cancels_then_cleans(self):
        self.provider.block = True
        errors = []
        def run():
            try:
                self.runtime.generate(self.request, self.cancel)
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=run)
        worker.start()
        self.assertTrue(self.provider.started.wait(2))
        try:
            with self.assertRaises(ResourceExhausted):
                self.resources.reserve('image', 'image', 800, {0: 800})
            with self.assertRaises(ResourceBusy):
                self.runtime.generate(self.request, threading.Event())
            self.assertEqual(self.provider.unloads, 0)
            self.runtime.unload()
        finally:
            self.cancel.set()
            worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(errors[0], ResourceCancelled)
        self.assertEqual(self.resources.snapshot()['reservations'], {})

    def test_cancel_before_loading_does_not_allocate(self):
        self.cancel.set()
        with self.assertRaises(ResourceCancelled):
            self.runtime.generate(self.request, self.cancel)
        self.assertEqual(self.provider.loads, 0)

    def test_failed_admission_does_not_load(self):
        other = self.resources.reserve('busy', 'llm', 800, {0: 800})
        try:
            with self.assertRaises(ResourceExhausted):
                self.runtime.generate(self.request, self.cancel)
            self.assertEqual(self.provider.loads, 0)
        finally:
            other.release()

    def test_replacing_provider_only_requires_registration(self):
        app = FastAPI()
        app.include_router(router)
        manager = MemoryManager()
        manager.tts = SpeechRequests(self.runtime)
        manager.request_executors['tts'] = manager.tts
        with patch.object(speech, 'speech_runtime', self.runtime), patch.object(
                memory_manager, 'submit', direct_feature(manager, 'tts')), TestClient(app) as client:
            models = client.get('/v1/audio/speech/models').json()
            self.assertEqual(models, [{'id': 'alternate', 'name': 'Alternative voice',
                'mode': 'custom', 'speakers': ['Ada'], 'supports_instruction': True, 'default_speaker': None}])
            for _ in range(2):
                response = client.post('/v1/audio/speech', json={'script': 'Hello', 'language': 'Martian',
                    'voice': {'mode': 'custom', 'speaker': 'Ada', 'instruction': 'warm'}})
                self.assertEqual(response.status_code, 200, response.text)
                audio = response.json()['audio']
                self.assertEqual(audio['mime_type'], 'audio/wav')
                self.assertEqual(base64.b64decode(audio['audio_base64']), wav_bytes())
                self.assertEqual(audio['voice'], 'alternate')
            self.assertEqual(self.provider.loads, 1)
            response = client.post('/v1/audio/speech', json={'script': 'Hello',
                'voice': {'mode': 'custom', 'speaker': 'Wrong'}})
            self.assertEqual(response.status_code, 422)
            self.assertEqual(self.provider.calls, 2)

    def test_provider_identity_prevents_accidental_resident_reuse(self):
        second = AlternateProvider(self.resources, 'different')
        self.registry.register(second)
        self.runtime.generate(self.request, self.cancel)
        request = SpeechInput('Hello', self.request.voice, model_id='different')
        self.runtime.generate(request, self.cancel)
        self.assertEqual((self.provider.unloads, second.loads), (1, 1))

    def test_duplicate_registration_is_rejected(self):
        with self.assertRaises(ValueError):
            self.registry.register(self.provider)

    def test_two_runtimes_share_one_process_resource_manager(self):
        self.resources = ResourceManager(600, {0: 600})
        self.provider.resources = self.resources
        other = SpeechRuntime(self.registry, lambda: self.resources)
        self.runtime.load(self.request, self.cancel)
        other.load(self.request, self.cancel)
        self.assertEqual(len(self.resources.snapshot()['reservations']), 2)
        self.assertEqual((self.provider.loads, self.provider.unloads), (2, 1))
        other.unload()

    def test_busy_eviction_callback_does_not_deadlock_admission(self):
        self.runtime.load(self.request, self.cancel)
        with self.runtime._gate:
            with self.assertRaises(ResourceExhausted):
                self.resources.reserve('image', 'image', 800, {0: 800})
        self.provider.assert_owned(active=False)
