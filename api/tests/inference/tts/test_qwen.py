import unittest
from api.inference.tts.qwen import QwenSpeechProvider
from api.inference.tts.speech_runtime import SpeechInput, SpeechUnavailable


class NativeQwenCapabilitiesTests(unittest.TestCase):
    def test_only_executable_native_voice_is_advertised(self):
        provider=QwenSpeechProvider();models=provider.models()
        self.assertEqual([m.id for m in models],['qwen-tts-1.7b-custom'])
        self.assertEqual(models[0].speakers,('Ryan',))
        self.assertFalse(models[0].supports_instruction)
        self.assertFalse(provider.enabled('qwen-tts-1.7b-design'))
    def test_unsupported_voice_language_and_model_fail_closed(self):
        provider=QwenSpeechProvider()
        for voice,language,model in [({'mode':'describe','description':'warm'},'English','qwen-tts-1.7b-design'),
                ({'mode':'custom','speaker':'Ryan'},'French','qwen-tts-1.7b-custom'),
                ({'mode':'custom','speaker':'Ryan','instruction':'warm'},'English','qwen-tts-1.7b-custom'),
                ({'mode':'clone','sample':'abc'},'English','qwen-tts-1.7b-base')]:
            with self.subTest(voice=voice),self.assertRaises(ValueError):
                provider.validate(SpeechInput('Hello',voice,language,model))
