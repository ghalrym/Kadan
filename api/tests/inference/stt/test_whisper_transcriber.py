import base64
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import wave
from api.inference.errors import InferenceFailure
from api.inference.stt.native_worker import NativeWhisper
from api.inference.stt.whisper_transcriber import WhisperTranscriber, get_whisper_transcriber, _cached_whisper_transcriber
from api.inference.stt.catalog import OFFICIAL_CHECKPOINTS, checkpoint


def audio_url():
    data = io.BytesIO()
    with wave.open(data, 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(b'\x00\x00' * 160)
    return 'data:audio/wav;base64,' + base64.b64encode(data.getvalue()).decode()


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.worker = Mock()
        self.store = Mock(root=Path(self.temp.name))
        self.manager = WhisperTranscriber(worker=self.worker, store=self.store)
    def test_default_is_native_without_opt_in(self):
        with patch.dict('os.environ', {}, clear=True):
            self.assertIsInstance(WhisperTranscriber()._cpp, NativeWhisper)
    def test_aliases_and_selection_preserved_without_loading(self):
        self.assertEqual(len(OFFICIAL_CHECKPOINTS), 12)
        self.assertEqual(self.manager.select('large'), 'large-v3')
        self.assertEqual(self.manager.selected(), 'large-v3')
        self.worker.load.assert_not_called()
    def test_native_request_and_residency_lifecycle(self):
        self.worker.transcribe.return_value = {'text': 'hello', 'raw_text': 'hello'}
        audio = audio_url()
        result = self.manager.transcribe(audio, 'large', 'en')
        self.assertEqual(result['text'], 'hello')
        self.worker.transcribe.assert_called_once_with(audio, 'large-v3', 'en', None)
        self.manager.offload_to_ram()
        self.worker.offload_to_ram.assert_called_once_with(None)
        self.assertIs(self.manager.whisper_model, self.worker)
        self.manager.close()
        self.worker.close.assert_called_once()
        self.assertIsNone(self.manager.whisper_model)
    def test_native_failure_propagates_without_python_fallback(self):
        self.worker.transcribe.side_effect = RuntimeError('native failure')
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            self.manager.transcribe(audio_url(), 'large-v3')
        self.assertIs(self.manager._cpp, self.worker)
    def test_failed_cleanup_preserves_worker_handle(self):
        self.manager.load('large-v3')
        self.worker.close.side_effect = RuntimeError('reap failed')
        with self.assertRaisesRegex(RuntimeError, 'reap failed'):
            self.manager.close()
        self.assertIs(self.manager.whisper_model, self.worker)
    def test_disabled_model_rejected_before_worker_execution(self):
        with patch('api.inference.stt.whisper_transcriber.get_whisper_checkpoints', return_value={}):
            with self.assertRaises(InferenceFailure):
                self.manager.transcribe(audio_url(), 'large-v3')
        self.worker.transcribe.assert_not_called()
