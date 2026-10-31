import unittest

from pydantic import ValidationError

from api.pydantic_models.model_store import ModelAssignments, ModelStoreStatus


class ModelAssignmentsTests(unittest.TestCase):
    def test_machine_resources_cannot_change_assignments(self):
        for field in ModelAssignments.model_fields:
            with self.subTest(field=field), self.assertRaises(ValidationError):
                ModelAssignments(**{field: "another-model"})

    def test_new_store_has_fixed_assignments_without_a_saved_selection(self):
        status = ModelStoreStatus(models=[])
        self.assertEqual(status.defaults.transcription, "whisper-medium")
        self.assertEqual(status.defaults.formatting, "superwhisper-s1-mini")
        self.assertEqual(status.defaults.speech, "qwen-tts-1.7b-custom")
        self.assertEqual(status.selected_model_id, status.defaults.llm)
