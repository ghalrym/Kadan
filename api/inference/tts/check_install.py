"""Import native speech coordination without loading Python inference models."""
from api.inference.tts.native_worker import NativeSpeechSession
from api.inference.tts.qwen import QwenSpeechProvider
from api.inference.stt.native_worker import NativeWhisper
from api.inference.stt.whisper_transcriber import WhisperTranscriber


def check_speech_install():
    assert QwenSpeechProvider().models()[0].id == 'qwen-tts-1.7b-custom'
    assert isinstance(WhisperTranscriber()._cpp, NativeWhisper)
    assert NativeSpeechSession
    print('Native speech worker adapters import successfully; no model execution.')


if __name__ == '__main__':
    check_speech_install()
