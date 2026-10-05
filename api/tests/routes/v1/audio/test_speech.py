import unittest
from unittest.mock import patch

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
        disabled = patch("api.services.speech.ENABLED_SPEECH_MODELS", frozenset())
        disabled.start()
        self.addCleanup(disabled.stop)

    def test_history_is_empty_and_defaults_are_blank(self):
        response = self.client.get('/v1/audio/speech')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'audio': [], 'voice_description': '', 'script': ''})

    def test_both_modes_report_unavailable_without_fake_audio(self):
        for voice in [{'mode': 'describe', 'description': 'Warm'}, {'mode': 'clone', 'sample': 'UklGRg==', 'speaker_only': True}]:
            with self.subTest(voice=voice):
                response = self.client.post('/v1/audio/speech', json={'script': 'Hello', 'voice': voice})
                self.assertEqual(response.status_code, 503)
                self.assertIn('not enabled', response.json()['detail'])
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

    def test_long_script_and_voice_reach_provider_without_arbitrary_caps(self):
        for voice in [
            {'mode': 'describe', 'description': 'v' * 10000},
            {'mode': 'clone', 'sample': 'c3Nz' * 10000, 'speaker_only': True},
        ]:
            with self.subTest(mode=voice['mode']):
                response = self.client.post('/v1/audio/speech', json={'script': 'x' * 20000, 'voice': voice})
                self.assertEqual(response.status_code, 503)
                self.assertIn('not enabled', response.json()['detail'])

    def test_mode_mismatch_and_small_custom_instructions_are_rejected(self):
        for body in [
            {'script': 'hello', 'model_id': 'qwen-tts-1.7b-base', 'voice': {'mode': 'describe', 'description': 'warm'}},
            {'script': 'hello', 'model_id': 'qwen-tts-0.6b-custom', 'voice': {'mode': 'custom', 'speaker': 'Ryan', 'instruction': 'excited'}},
            {'script': 'hello', 'voice': {'mode': 'clone', 'sample': 'https://example.org/audio.wav', 'speaker_only': True}},
        ]:
            self.assertEqual(self.client.post('/v1/audio/speech', json=body).status_code, 422)
