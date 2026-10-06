"""Official Qwen3-TTS checkpoints; artifact revisions are immutable."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SpeechModel:
    id: str
    name: str
    revision: str
    estimated_bytes: int
    mode: str


SPEECH_MODELS = {
    item.id: item for item in (
        SpeechModel('qwen-tts-1.7b-custom', 'Qwen3-TTS-12Hz-1.7B-CustomVoice', '0c0e3051f131929182e2c023b9537f8b1c68adfe', 4_520_000_000, 'custom'),
        SpeechModel('qwen-tts-0.6b-custom', 'Qwen3-TTS-12Hz-0.6B-CustomVoice', '85e237c12c027371202489a0ec509ded67b5e4b5', 2_500_000_000, 'custom'),
        SpeechModel('qwen-tts-1.7b-design', 'Qwen3-TTS-12Hz-1.7B-VoiceDesign', '5ecdb67327fd37bb2e042aab12ff7391903235d3', 4_520_000_000, 'describe'),
        SpeechModel('qwen-tts-1.7b-base', 'Qwen3-TTS-12Hz-1.7B-Base', 'fd4b254389122332181a7c3db7f27e918eec64e3', 4_540_000_000, 'clone'),
        SpeechModel('qwen-tts-0.6b-base', 'Qwen3-TTS-12Hz-0.6B-Base', '5d83992436eae1d760afd27aff78a71d676296fc', 2_520_000_000, 'clone'),
    )
}
SPEAKERS = ('Vivian', 'Serena', 'Uncle_Fu', 'Dylan', 'Eric', 'Ryan', 'Aiden', 'Ono_Anna', 'Sohee')
LANGUAGES = ('Auto', 'Chinese', 'English', 'Japanese', 'Korean', 'German', 'French', 'Russian', 'Portuguese', 'Spanish', 'Italian')
