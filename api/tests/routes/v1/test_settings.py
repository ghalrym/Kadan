import unittest

from pydantic import ValidationError

from api.routes.v1.settings import SettingsRequest


class SettingsValidationTests(unittest.TestCase):
    def test_settings_reject_model_from_another_category(self):
        with self.assertRaises(ValidationError):
            SettingsRequest(models={"LLM": "F5-TTS"})
