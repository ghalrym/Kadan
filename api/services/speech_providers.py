"""Speech adapter composition; checkpoint PRs enable models independently."""
from api.inference.speech import SpeechRegistry
from api.inference.qwen_speech import QwenSpeechProvider

speech_registry = SpeechRegistry()
speech_registry.register(QwenSpeechProvider())
