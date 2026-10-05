import unittest
from api.services.model_catalog import CATALOG, allowed_asset, validate_assets
from api.routes.v1.audio.speech import SpeechRequest, list_speech_models
from api.services.qwen_tts_catalog import SPEECH_MODELS
from api.services.speech_enabled import ENABLED_SPEECH_MODELS


class CheckpointTests(unittest.TestCase):
    def test_official_checkpoint_and_mode_are_exposed(self):
        model_id = 'qwen-tts-0.6b-base'
        self.assertIn(model_id, ENABLED_SPEECH_MODELS)
        model = SPEECH_MODELS[model_id]
        self.assertEqual(len(model.revision), 40)
        self.assertEqual(model.mode, 'clone')
        self.assertIn(model_id, [item.id for item in list_speech_models()])
        request = SpeechRequest(script='Hello', model_id=model_id, voice={'mode': 'clone', 'sample': 'UklGRg==', 'speaker_only': True})
        self.assertEqual(request.voice.mode, 'clone')

    def test_download_manifest_requires_both_weights(self):
        entry = CATALOG['qwen-tts-0.6b-base']
        assets = set(entry.required_files)
        self.assertTrue(all(allowed_asset(name, entry) for name in assets))
        validate_assets(entry, assets)
        for missing in ('model.safetensors', 'speech_tokenizer/model.safetensors'):
            with self.assertRaises(ValueError):
                validate_assets(entry, assets - {missing})
        self.assertFalse(allowed_asset('speech_tokenizer/run.py', entry))
