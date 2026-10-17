import unittest

from pydantic import ValidationError

from api.pydantic_models.model_store import TierModels


class TierModelsTests(unittest.TestCase):
    def test_language_model_is_required(self):
        for assignments in ({}, {"llm": None}):
            with (
                self.subTest(assignments=assignments),
                self.assertRaises(ValidationError),
            ):
                TierModels.model_validate(assignments)

    def test_unassigned_modalities_serialize_as_null(self):
        models = TierModels(llm="language-model")
        self.assertEqual(
            models.model_dump(),
            {
                "llm": "language-model",
                "image": None,
                "video": None,
                "speech": None,
                "transcription": None,
                "formatting": None,
            },
        )
