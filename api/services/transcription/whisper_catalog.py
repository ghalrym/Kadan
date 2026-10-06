"""Official OpenAI Whisper checkpoints; aliases never create duplicate storage."""
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path


@dataclass(frozen=True)
class WhisperCheckpoint:
    name: str
    sha256: str
    memory_gib: int
    device_memory_gib: int

    @property
    def url(self):
        return f"https://openaipublic.azureedge.net/main/whisper/models/{self.sha256}/{self.name}.pt"


# CPU estimates include checkpoint construction. Device estimates add headroom
# over OpenAI's approximate 1/1/2/5/10/6 GB tiny/base/small/medium/large/turbo table:
# https://github.com/openai/whisper#available-models-and-languages
OFFICIAL_CHECKPOINTS = {
    'tiny.en': WhisperCheckpoint('tiny.en', 'd3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03', 2, 2),
    'tiny': WhisperCheckpoint('tiny', '65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9', 2, 2),
    'base.en': WhisperCheckpoint('base.en', '25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead', 2, 2),
    'base': WhisperCheckpoint('base', 'ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e', 2, 2),
    'small.en': WhisperCheckpoint('small.en', 'f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872', 4, 3),
    'small': WhisperCheckpoint('small', '9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794', 4, 3),
    'medium.en': WhisperCheckpoint('medium.en', 'd7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f', 10, 6),
    'medium': WhisperCheckpoint('medium', '345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1', 10, 6),
    'large-v1': WhisperCheckpoint('large-v1', 'e4b87e7e0bf463eb8e6956e646f1e277e901512310def2c24bf0e11bd3c28e9a', 20, 12),
    'large-v2': WhisperCheckpoint('large-v2', '81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524', 20, 12),
    'large-v3': WhisperCheckpoint('large-v3', 'e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb', 20, 12),
    'large-v3-turbo': WhisperCheckpoint('large-v3-turbo', 'aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a', 12, 8),
}
ALIASES = {"large": "large-v3", "turbo": "large-v3-turbo"}


def checkpoint(name):
    """Resolve official aliases or reject unknown model names."""
    try:
        return OFFICIAL_CHECKPOINTS[ALIASES.get(name, name)]
    except KeyError:
        raise ValueError("Unknown Whisper checkpoint") from None


@lru_cache(maxsize=1)
def get_whisper_checkpoints():
    """Enable only checkpoint registrations shipped with this version of Kadan."""
    enabled = {}
    for path in sorted((Path(__file__).parents[1] / "whisper_checkpoints").glob("*.json")):
        name = json.loads(path.read_text())["name"]
        entry = checkpoint(name)
        if entry.name != name:
            raise ValueError("Whisper registrations must use canonical names")
        enabled[name] = entry
    return enabled

