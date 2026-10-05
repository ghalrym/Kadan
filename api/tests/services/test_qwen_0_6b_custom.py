import unittest
from api.routes.v1.audio.speech import SpeechRequest, list_speech_models
from api.services.qwen_tts_catalog import SPEECH_MODELS
from api.services.speech_enabled import ENABLED_SPEECH_MODELS


class CheckpointTests(unittest.TestCase):
    def test_official_checkpoint_and_mode_are_exposed(self):
        model_id = 'qwen-tts-0.6b-custom'
        self.assertIn(model_id, ENABLED_SPEECH_MODELS)
        model = SPEECH_MODELS[model_id]
        self.assertEqual(len(model.revision), 40)
        self.assertEqual(model.mode, 'custom')
        self.assertIn(model_id, [item.id for item in list_speech_models()])
        request = SpeechRequest(script='Hello', model_id=model_id, voice={'mode': 'custom', 'speaker': 'Ryan'})
        self.assertEqual(request.voice.mode, 'custom')
