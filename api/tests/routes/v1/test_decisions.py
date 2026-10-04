import unittest

from pydantic import ValidationError

from api.pydantic_models.decisions import ScoreQuestion
from api.routes.v1.decisions import DecisionRequest


class DecisionValidationTests(unittest.TestCase):
    def test_decision_keys_must_be_unique(self):
        with self.assertRaises(ValidationError):
            DecisionRequest(state="Incident", questions=[ScoreQuestion(
                key="severity", instructions="Rate severity", type="Score", levels=["Low", "High"],
            )] * 2)

    def test_question_type_requires_its_own_fields(self):
        with self.assertRaises(ValidationError):
            DecisionRequest(state="Incident", questions=[{
                "key": "severity", "instructions": "Rate severity", "type": "Score",
                "options": [{"key": "high", "description": "High"}],
            }])
