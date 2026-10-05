import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api.server import app
from api.services.transcript_formatting import FormattedTranscript
from api.services.runtime import RuntimeFailure
from api.services.whisper_catalog import CHECKPOINTS


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_references_are_not_fetched(self):
        with patch('api.services.transcription.CHECKPOINTS', {'tiny': object()}):
            response = self.client.post('/v1/audio/transcriptions', json={'audio': 'https://example.com/audio.wav'})
        self.assertEqual(response.status_code, 422)
        self.assertIn('not fetched', response.json()['detail'])

    def test_invalid_input_and_models(self):
        for body in ({}, {'audio': ''}, {'audio': '   '}, {'audio': 'ref', 'file': 'fake'}, {'audio': 'ref', 'model': 'distil'}):
            self.assertEqual(self.client.post('/v1/audio/transcriptions', json=body).status_code, 422)

    def test_native_output_and_raw_transcript(self):
        output = dict(text='hello', raw_text='hello', language='en', model='tiny', formatting_status='disabled')
        with patch('api.routes.v1.audio.transcriptions.transcription_manager.transcribe', return_value=output) as native:
            response = self.client.post('/v1/audio/transcriptions', json={'audio': 'data', 'model': 'turbo', 'formatting': False})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['raw_text'], 'hello')
        native.assert_called_once_with('data', 'large-v3-turbo', None)

    def test_unavailable_and_busy_are_not_transcripts(self):
        for status in (409, 503):
            with patch('api.routes.v1.audio.transcriptions.transcription_manager.transcribe', side_effect=RuntimeFailure('Not ready', status)):
                response = self.client.post('/v1/audio/transcriptions', json={'audio': 'data'})
            self.assertEqual(response.status_code, status)
            self.assertNotIn('text', response.json())

    def test_unique_catalog(self):
        response = self.client.get('/v1/audio/transcriptions/models')
        self.assertEqual(response.json()['models'], list(CHECKPOINTS))
        self.assertNotIn('large', response.json()['models'])
        self.assertNotIn('turbo', response.json()['models'])

    def test_s1_formats_after_asr_and_keeps_raw_text(self):
        output = dict(text='um hello', raw_text='um hello', language='en', model='tiny')
        with patch('api.routes.v1.audio.transcriptions.transcription_manager.transcribe', return_value=output), patch(
                'api.routes.v1.audio.transcriptions.format_transcript', return_value=FormattedTranscript(
                    'um hello', 'Hello.', 'formatted', 'S1-mini by Superwhisper')) as formatter:
            response = self.client.post('/v1/audio/transcriptions', json={'audio': 'data', 'formatting': True})
        formatter.assert_called_once_with('um hello', 'en', True)
        self.assertEqual(response.json()['text'], 'Hello.')
        self.assertEqual(response.json()['raw_text'], 'um hello')
        self.assertEqual(response.json()['formatting_model'], 'S1-mini by Superwhisper')
