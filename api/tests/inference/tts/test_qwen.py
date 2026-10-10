import unittest
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from api.inference.tts.qwen import QwenSpeechProvider
from api.inference.tts.catalog import SPEAKERS, LANGUAGES, SPEECH_MODELS
from api.inference.resources import ResourceCancelled
from api.inference.tts.speech_runtime import SpeechInput, SpeechUnavailable


class NativeQwenCapabilitiesTests(unittest.TestCase):
    def test_acquisition_precedes_export_and_receives_request_cancellation(self):
        request = SimpleNamespace(model_id='qwen-tts-1.7b-custom')
        cancel = threading.Event()
        entry = SPEECH_MODELS[request.model_id]
        with patch.dict('os.environ', {}, clear=True), \
                patch('api.inference.tts.qwen.model_manager') as manager, \
                patch('api.inference.tts.qwen.ensure_assets') as export:
            manager.root = Path('/models')
            manager.ensure_checkpoint.return_value = entry, Path('/checkpoint')
            QwenSpeechProvider().prepare_assets(request, 'resources', cancel)
            manager.ensure_checkpoint.assert_called_once_with(request.model_id, cancel)
            export.assert_called_once_with('tts', Path('/checkpoint'), Path('/models/native/tts'), 'resources', cancel)
            export.reset_mock()
            manager.ensure_checkpoint.side_effect = ResourceCancelled('cancelled')
            with self.assertRaises(ResourceCancelled):
                QwenSpeechProvider().prepare_assets(request, 'resources', cancel)
            export.assert_not_called()

    def test_custom_tokenizer_still_acquires_model_weights(self):
        request = SimpleNamespace(model_id='qwen-tts-1.7b-custom')
        cancel = threading.Event()
        with patch.dict('os.environ', {'KADAN_NATIVE_TTS_TOKENIZER': '/tokenizer'}), \
                patch('api.inference.tts.qwen.model_manager') as manager, \
                patch('api.inference.tts.qwen.ensure_assets') as export:
            manager.ensure_checkpoint.return_value = SPEECH_MODELS[request.model_id], Path('/checkpoint')
            QwenSpeechProvider().prepare_assets(request, None, cancel)
            manager.ensure_checkpoint.assert_called_once_with(request.model_id, cancel)
            export.assert_not_called()

    def test_only_executable_native_voice_is_advertised(self):
        provider=QwenSpeechProvider();models=provider.models()
        self.assertEqual([m.id for m in models],['qwen-tts-1.7b-custom'])
        self.assertEqual(models[0].speakers,SPEAKERS)
        self.assertTrue(models[0].supports_instruction)
        self.assertFalse(provider.enabled('qwen-tts-1.7b-design'))
    def test_unsupported_voice_language_and_model_fail_closed(self):
        provider=QwenSpeechProvider()
        for voice,language,model in [({'mode':'describe','description':'warm'},'English','qwen-tts-1.7b-design'),
                ({'mode':'custom','speaker':'Ryan'},'Klingon','qwen-tts-1.7b-custom'),
                ({'mode':'custom','speaker':'Ryan','instruction':'x'*8001},'English','qwen-tts-1.7b-custom'),
                ({'mode':'clone','sample':'abc'},'English','qwen-tts-1.7b-base')]:
            with self.subTest(voice=voice),self.assertRaises(ValueError):
                provider.validate(SpeechInput('Hello',voice,language,model))

    def test_all_advertised_speakers_and_languages_validate(self):
        provider=QwenSpeechProvider()
        for speaker in SPEAKERS:
            for language in LANGUAGES:
                provider.validate(SpeechInput('Hello',dict(mode='custom',speaker=speaker,instruction='Warmly'),language,'qwen-tts-1.7b-custom'))
