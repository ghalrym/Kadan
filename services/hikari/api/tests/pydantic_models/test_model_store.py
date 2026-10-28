import unittest

from pydantic import ValidationError

from api.pydantic_models.model_store import ModelAssignments, ModelStoreStatus


class FixedAssignmentsTests(unittest.TestCase):
    def test_assignments_are_fixed_and_do_not_imply_checkpoint_availability(self):
        status = ModelStoreStatus(models=[])
        self.assertEqual(status.defaults.llm, "lfm3-1.7b-bf16")
        self.assertEqual(status.defaults.speech, "lfm3-audio-2b-bf16")
        self.assertEqual(status.defaults.image, "qwen-image-2.1-turbo")
        self.assertEqual(status.models, [])
        self.assertNotIn("tiers", status.model_dump())

    def test_alternative_models_cannot_change_the_fixed_stack(self):
        for name, value in (("llm", "qwen"), ("speech", "qwen-tts"), ("image", None)):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                ModelAssignments.model_validate({name: value})
