import unittest

from pydantic import ValidationError

from api.routes.v1.audio.speech import SpeechRequest


class SpeechValidationTests(unittest.TestCase):
    def test_clone_requires_sample(self):
        with self.assertRaises(ValidationError):
            SpeechRequest(script="Hello", voice={"mode": "clone", "description": "Warm"})
