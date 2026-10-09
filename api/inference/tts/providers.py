"""Speech adapter composition; checkpoint PRs enable models independently."""
from api.inference.tts.speech_runtime import SpeechRegistry
from api.inference.tts.qwen import QwenSpeechProvider

speech_registry = SpeechRegistry()
speech_registry.register(QwenSpeechProvider())
