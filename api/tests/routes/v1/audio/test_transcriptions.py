import unittest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from api.server import app
from api.services.runtime import RuntimeFailure
from api.services.transcription.whisper_catalog import get_whisper_checkpoints


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_references_are_not_fetched(self):
        with patch('api.services.transcription.transcription.get_whisper_checkpoints', return_value={'tiny': object()}):
            response = self.client.post('/v1/audio/transcriptions', json={'audio': 'https://example.com/audio.wav'})
        self.assertEqual(response.status_code, 422)
        self.assertIn('not fetched', str(response.json()['detail']))

    def test_invalid_input_and_models(self):
        for body in ({}, {'audio': ''}, {'audio': '   '}, {'audio': 'ref', 'file': 'fake'}, {'audio': 'ref', 'model': 'distil'}):
            self.assertEqual(self.client.post('/v1/audio/transcriptions', json=body).status_code, 422)

    def test_native_output_and_raw_transcript(self):
        output = dict(text='hello', raw_text='hello', language='en', model='tiny', formatting_status='disabled')
        with patch('api.routes.v1.audio.transcriptions.memory_manager.stt', AsyncMock(return_value=output)) as native:
            response = self.client.post('/v1/audio/transcriptions', json={'audio': 'data:audio/wav;base64,fixture', 'model': 'turbo', 'formatting': False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['raw_text'], 'hello')
        self.assertEqual(native.call_args.args[0].model, 'large-v3-turbo')

    def test_unavailable_and_busy_are_not_transcripts(self):
        for status in (409, 503):
            with patch('api.routes.v1.audio.transcriptions.memory_manager.stt', AsyncMock(side_effect=RuntimeFailure('Not ready', status))):
                response = self.client.post('/v1/audio/transcriptions', json={'audio': 'data:audio/wav;base64,fixture'})
            self.assertEqual(response.status_code, status)
            self.assertNotIn('text', response.json())

    def test_unique_catalog(self):
        response = self.client.get('/v1/audio/transcriptions/models')
        self.assertEqual(response.json()['models'], list(get_whisper_checkpoints()))
        self.assertNotIn('large', response.json()['models'])
        self.assertNotIn('turbo', response.json()['models'])
