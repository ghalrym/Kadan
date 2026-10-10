from unittest.mock import patch, AsyncMock
from api.memory_manager import memory_manager
from api.tests.memory_manager.helpers import direct_feature
import unittest
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.routes.v1.audio.speech import SpeechRequest, router


class SpeechValidationTests(unittest.TestCase):
    def test_clone_requires_sample(self):
        with self.assertRaises(ValidationError):
            SpeechRequest(script="Hello", voice={"mode": "clone", "description": "Warm"})


class SpeechRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(router)
        cls.client = TestClient(app)

    def setUp(self):
        self.enterContext(patch.object(memory_manager, 'submit', direct_feature(memory_manager, 'tts')))
        disabled = patch("api.inference.tts.enabled.ENABLED_SPEECH_MODELS", frozenset())
        disabled.start()
        self.addCleanup(disabled.stop)

    def test_history_is_empty_and_defaults_are_blank(self):
        response = self.client.get('/v1/audio/speech')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'audio': [], 'voice_description': '', 'script': ''})

    def test_unsupported_modes_reject_without_fake_audio(self):
        for voice in [{'mode': 'describe', 'description': 'Warm'}, {'mode': 'clone', 'sample': 'UklGRg==', 'speaker_only': True}]:
            with self.subTest(voice=voice):
                response = self.client.post('/v1/audio/speech', json={'script': 'Hello', 'voice': voice})
                self.assertEqual(response.status_code, 422)
                self.assertTrue(response.json()['detail'])
                self.assertNotIn('audio', response.json())
        self.assertEqual(self.client.get('/v1/audio/speech').json()['audio'], [])

    def test_invalid_modes_and_blank_fields_fail_before_provider(self):
        invalid = [
            {'script': 'Hello', 'voice': {'mode': 'clone', 'description': 'Warm'}},
            {'script': 'Hello', 'voice': {'mode': 'describe', 'sample': 'UklGRg==', 'speaker_only': True}},
            {'script': 'Hello', 'voice': {'mode': 'other', 'description': 'Warm'}},
            {'script': ' ', 'voice': {'mode': 'describe', 'description': 'Warm'}},
            {'script': 'Hello', 'voice': {'mode': 'clone', 'sample': ' '}},
            {'script': 'Hello', 'voice': {'mode': 'describe', 'description': 'Warm', 'sample': 'other'}},
        ]
        for body in invalid:
            with self.subTest(body=body):
                self.assertEqual(self.client.post('/v1/audio/speech', json=body).status_code, 422)

    def test_unsupported_long_voice_requests_reject_before_provider(self):
        for voice in [
            {'mode': 'describe', 'description': 'v' * 10000},
            {'mode': 'clone', 'sample': 'c3Nz' * 10000, 'speaker_only': True},
        ]:
            with self.subTest(mode=voice['mode']):
                response = self.client.post('/v1/audio/speech', json={'script': 'x' * 20000, 'voice': voice})
                self.assertEqual(response.status_code, 422)
                self.assertTrue(response.json()['detail'])

    def test_mode_mismatch_and_small_custom_instructions_are_rejected(self):
        for body in [
            {'script': 'hello', 'model_id': 'qwen-tts-1.7b-base', 'voice': {'mode': 'describe', 'description': 'warm'}},
            {'script': 'hello', 'model_id': 'qwen-tts-0.6b-custom', 'voice': {'mode': 'custom', 'speaker': 'Ryan', 'instruction': 'excited'}},
            {'script': 'hello', 'voice': {'mode': 'clone', 'sample': 'https://example.org/audio.wav', 'speaker_only': True}},
        ]:
            self.assertEqual(self.client.post('/v1/audio/speech', json=body).status_code, 422)


class NativeDefaultSpeechRouteTests(unittest.TestCase):
    def setUp(self):
        app=FastAPI();app.include_router(router)
        self.client=TestClient(app,raise_server_exceptions=False)
        self.enterContext(patch.dict(os.environ,{},clear=True))
        self.submit=self.enterContext(patch.object(memory_manager,'submit',new_callable=AsyncMock))
        self.submit.return_value=dict(voice='Ryan',meta='WAV',script='Hello',time='1s',audio_base64='UklGRg==',mime_type='audio/wav')
        self.body={'script':'Hello','voice':{'mode':'custom','speaker':'Ryan'}}
    def test_omitted_language_and_explicit_english_enter_queue(self):
        for extra in ({},{'language':'English'}):
            with self.subTest(extra=extra):
                self.assertEqual(self.client.post('/v1/audio/speech',json={**self.body,**extra}).status_code,200)
        self.assertEqual(self.submit.await_count,2)
    def test_unsupported_or_oversized_input_is_422_before_queue(self):
        for extra in ({'voice':{'mode':'custom','speaker':'Vivian'}},{'language':'French'},
                      {'script':'x'*32001},{'script':'é'*16001}):
            with self.subTest(extra=list(extra)):
                response=self.client.post('/v1/audio/speech',json={**self.body,**extra})
                self.assertEqual(response.status_code,422,response.text)
        self.submit.assert_not_awaited()
