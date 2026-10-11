import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from api.inference.tts.native_worker import NativeSpeechSession, PROCESS_BUDGET
from api.inference.tts.qwen import QwenSpeechProvider
from api.inference.tts.speech_runtime import SpeechInput, SpeechUnavailable
from api.inference.tts.speech_requests import SpeechRequests
from api.inference.stt.transcription_requests import TranscriptionRequests


class NativeProviderTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        worker = root / 'worker'
        tokens = root / 'tokens'
        worker.touch()
        tokens.touch()
        env = patch.dict(os.environ, {'KADAN_NATIVE_TTS_WORKER': str(worker), 'KADAN_NATIVE_TTS_TOKENIZER': str(tokens), 'KADAN_TTS_DEVICES': 'cpu'})
        env.start()
        self.addCleanup(env.stop)
        resolver = patch('api.inference.tts.qwen.model_manager.get_checkpoint', return_value=(SimpleNamespace(revision='0c0e3051f131929182e2c023b9537f8b1c68adfe'), root))
        resolver.start()
        self.addCleanup(resolver.stop)
        self.request = SpeechInput('Hello.', {'mode': 'custom', 'speaker': 'Ryan'}, 'English', 'qwen-tts-1.7b-custom')
        self.provider = QwenSpeechProvider()
    def test_native_selection_and_budget(self):
        with patch('api.inference.tts.qwen.verify_tokenizer') as verify:
            plan = self.provider.prepare(self.request)
        verify.assert_called_once()
        self.assertEqual(plan.host_bytes, PROCESS_BUDGET)
        self.assertEqual(plan.device_bytes, {})
        self.assertIsInstance(plan.create(), NativeSpeechSession)
        self.assertEqual(len(self.provider.models()), 1)
        self.assertTrue(self.provider.models()[0].supports_instruction)
    def test_missing_export_is_configuration_failure(self):
        with patch('api.inference.tts.qwen.verify_tokenizer', side_effect=FileNotFoundError()):
            with self.assertRaises(SpeechUnavailable):
                self.provider.prepare(self.request)
    def test_queue_checks_tts_quarantine(self):
        def failed():
            raise SpeechUnavailable('quarantined')
        wrapper = SpeechRequests(SimpleNamespace(_session=SimpleNamespace(check_execution_state=failed)))
        with self.assertRaisesRegex(SpeechUnavailable, 'quarantined'):
            wrapper.check_execution_state()
    def test_queue_checks_whisper_quarantine(self):
        def failed():
            raise RuntimeError('quarantined')
        wrapper = TranscriptionRequests(SimpleNamespace(_cpp=SimpleNamespace(check_execution_state=failed)))
        with self.assertRaisesRegex(RuntimeError, 'quarantined'):
            wrapper.check_execution_state()
