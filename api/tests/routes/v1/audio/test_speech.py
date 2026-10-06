from unittest.mock import patch
from api.memory_manager import memory_manager
from api.tests.memory_manager.helpers import direct_feature
import unittest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.routes.v1.audio.speech import SpeechRequest, router


class SpeechValidationTests(unittest.TestCase):
    def test_clone_requires_sample(self):
        with self.assertRaises(ValidationError):
            SpeechRequest(script="Hello", voice={"mode": "clone", "description": "Warm"})


class SpeechRouteTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(memory_manager, 'submit', direct_feature(memory_manager, 'tts')))

    @classmethod
    def setUpClass(cls):
        app = FastAPI()
        app.include_router(router)
        cls.client = TestClient(app)

    def test_history_is_empty_and_defaults_are_blank(self):
        response = self.client.get('/v1/audio/speech')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'audio': [], 'voice_description': '', 'script': ''})

    def test_both_modes_report_unavailable_without_fake_audio(self):
        for voice in [{'mode': 'describe', 'description': 'Warm'}, {'mode': 'clone', 'sample': 'sample-id'}]:
            with self.subTest(voice=voice):
                response = self.client.post('/v1/audio/speech', json={'script': 'Hello', 'voice': voice})
                self.assertEqual(response.status_code, 503)
                self.assertIn('No speech provider', response.json()['detail'])
                self.assertNotIn('audio', response.json())
        self.assertEqual(self.client.get('/v1/audio/speech').json()['audio'], [])

    def test_invalid_modes_and_blank_fields_fail_before_provider(self):
        invalid = [
            {'script': 'Hello', 'voice': {'mode': 'clone', 'description': 'Warm'}},
            {'script': 'Hello', 'voice': {'mode': 'describe', 'sample': 'sample-id'}},
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
            {'mode': 'clone', 'sample': 's' * 10000},
        ]:
            with self.subTest(mode=voice['mode']):
                response = self.client.post('/v1/audio/speech', json={'script': 'x' * 20000, 'voice': voice})
                self.assertEqual(response.status_code, 503)
                self.assertIn('No speech provider', response.json()['detail'])
