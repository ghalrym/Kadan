"""Speech choices documented by the pinned LFM3-Audio model card."""

from typing import Literal, get_args

SpeechLanguage = Literal[
    "Arabic",
    "Danish",
    "German",
    "Greek",
    "British English",
    "American English",
    "Finnish",
    "French",
    "Hebrew",
    "Italian",
    "Japanese",
    "Korean",
    "Dutch",
    "Polish",
    "Portuguese",
    "Spanish",
    "Swedish",
    "Turkish",
    "Chinese",
]
SpeechGender = Literal["female", "male"]
LANGUAGES = get_args(SpeechLanguage)
GENDERS = get_args(SpeechGender)
DEFAULT_LANGUAGE: SpeechLanguage = "American English"
DEFAULT_GENDER: SpeechGender = "female"
