import base64
import io
import json
import tempfile
from types import SimpleNamespace
from pathlib import Path
import threading
import struct
import unittest
from unittest.mock import Mock, patch
import wave

from api.inference.errors import InferenceFailure
from api.services.inference_requests import pcm_audio, prepare_request, prepare_chat
from api.routes.v1.audio.speech import SpeechRequest
from api.routes.v1.chat.completions import CompletionRequest
from api.routes.v1.videos.generations import VideoGenerationRequest
from pydantic import ValidationError


class InputBoundaryTests(unittest.TestCase):
    def test_unsupported_language_model_does_not_download(self):
        manager = Mock()
        manager._language_entry.return_value.inference_available = False
        with patch('api.services.inference_requests.model_manager', manager):
            with self.assertRaises(InferenceFailure):
                prepare_request('llm', 'completion', 'medium', Mock(), Path('/unused'), threading.Event())
        manager.ensure_checkpoint.assert_not_called()

    def test_wav_transport_decodes_pcm_without_model_execution(self):
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as source:
            source.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            source.writeframes(b'\x00\x00\x00\x40')
        result = pcm_audio('data:audio/wav;base64,' + base64.b64encode(buffer.getvalue()).decode())
        self.assertEqual(struct.unpack('<ff', result), (0.0, 0.5))

    def test_unsupported_video_settings_rejected_by_request_model(self):
        for value in ({'resolution': '1080p'}, {'duration': 120}, {'fps': 30}, {'negative_prompt': 'blur'}):
            with self.assertRaises(ValidationError):
                VideoGenerationRequest(prompt='test', **value)

    def test_supported_speech_contract_rejects_clone(self):
        with self.assertRaises(ValidationError):
            SpeechRequest(script='test', voice={'mode': 'clone', 'sample': 'data'})


class ChatTransportTests(unittest.TestCase):
    def test_history_is_written_without_image_character_limit_or_truncation(self):
        body = CompletionRequest(messages=[dict(role='user', text='a' * 50000)])
        entry = SimpleNamespace(id='small', inference_available=True)
        with tempfile.TemporaryDirectory() as folder, patch('api.services.inference_requests.model_manager') as manager:
            root = Path(folder)
            (root / 'config.json').write_text(json.dumps(dict(model_type='qwen3_5_moe', max_position_embeddings=262144)))
            manager.configured_context.return_value = 65536
            result = prepare_chat(entry, root, body, root, threading.Event(), 'completion')
            self.assertEqual(json.loads(Path(result.payload.input).read_text())['messages'][0]['text'], 'a' * 50000)
            self.assertEqual(result.payload.context_limit, 65536)
            self.assertLess(len(result.payload.model_dump_json()), 1024)
